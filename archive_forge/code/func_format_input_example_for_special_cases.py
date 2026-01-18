import json
import logging
import numpy as np
from mlflow.environment_variables import MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import _contains_params
from mlflow.types.schema import ColSpec, DataType, Schema, TensorSpec
from mlflow.utils.os import is_windows
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout
def format_input_example_for_special_cases(input_example, pipeline):
    """
    Handles special formatting for specific types of Pipelines so that the displayed example
    reflects the correct example input structure that mirrors the behavior of the input parsing
    for pyfunc.
    """
    import transformers
    input_data = input_example[0] if isinstance(input_example, tuple) else input_example
    if isinstance(pipeline, transformers.ZeroShotClassificationPipeline) and isinstance(input_data, dict) and isinstance(input_data['candidate_labels'], list):
        input_data['candidate_labels'] = json.dumps(input_data['candidate_labels'])
    return input_data if not isinstance(input_example, tuple) else (input_data, input_example[1])