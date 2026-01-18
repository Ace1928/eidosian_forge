import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from mlflow.exceptions import MlflowException
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.utils import _get_default_model, _get_latest_metric_version
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import (
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string
def eval_fn(predictions: 'pd.Series', metrics: Dict[str, MetricValue], inputs: 'pd.Series', *args) -> MetricValue:
    """
        This is the function that is called when the metric is evaluated.
        """
    eval_values = dict(zip(grading_context_columns, args))
    outputs = predictions.to_list()
    inputs = inputs.to_list()
    eval_model = evaluation_context['model']
    eval_parameters = evaluation_context['parameters']
    if not isinstance(eval_model, str):
        raise MlflowException(message=f'The model argument must be a string URI referring to an openai model (openai:/gpt-3.5-turbo) or an MLflow Deployments endpoint (endpoints:/my-endpoint), passed {eval_model} instead', error_code=INVALID_PARAMETER_VALUE)
    grading_payloads = []
    for indx, (input, output) in enumerate(zip(inputs, outputs)):
        try:
            arg_string = _format_args_string(grading_context_columns, eval_values, indx)
        except Exception as e:
            raise MlflowException(f"Values for grading_context_columns are malformed and cannot be formatted into a prompt for metric '{name}'.\nRequired columns: {grading_context_columns}\nValues: {eval_values}\nError: {e!r}\nPlease check the following: \n- predictions and targets (if required) are provided correctly\n- grading_context_columns are mapped correctly using the evaluator_config parameter\n- input and output data are formatted correctly.")
        grading_payloads.append(evaluation_context['eval_prompt'].format(input=input if include_input else None, output=output, grading_context_columns=arg_string))

    def score_model_on_one_payload(payload, eval_model):
        try:
            raw_result = model_utils.score_model_on_payload(eval_model, payload, eval_parameters)
            return _extract_score_and_justification(raw_result)
        except ImportError:
            raise
        except MlflowException as e:
            if e.error_code in [ErrorCode.Name(BAD_REQUEST), ErrorCode.Name(UNAUTHENTICATED), ErrorCode.Name(INVALID_PARAMETER_VALUE)]:
                raise
            else:
                return (None, f'Failed to score model on payload. Error: {e!s}')
        except Exception as e:
            return (None, f'Failed to score model on payload. Error: {e!s}')
    scores = [None] * len(inputs)
    justifications = [None] * len(inputs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_model_on_one_payload, payload, eval_model): indx for indx, payload in enumerate(grading_payloads)}
        as_comp = as_completed(futures)
        try:
            from tqdm.auto import tqdm
            as_comp = tqdm(as_comp, total=len(futures))
        except ImportError:
            pass
        for future in as_comp:
            indx = futures[future]
            score, justification = future.result()
            scores[indx] = score
            justifications[indx] = justification

    def aggregate_function(aggregate_option, scores):
        import numpy as np
        options = {'min': np.min, 'max': np.max, 'mean': np.mean, 'median': np.median, 'variance': np.var, 'p90': lambda x: np.percentile(x, 90) if x else None}
        if aggregate_option not in options:
            raise MlflowException(message=f'Invalid aggregate option {aggregate_option}.', error_code=INVALID_PARAMETER_VALUE)
        return options[aggregate_option](scores)
    scores_for_aggregation = [score for score in scores if score is not None]
    aggregate_results = {option: aggregate_function(option, scores_for_aggregation) for option in aggregations} if aggregations is not None else {}
    return MetricValue(scores, justifications, aggregate_results)