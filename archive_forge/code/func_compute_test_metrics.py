import collections
import concurrent.futures
import datetime
import itertools
import uuid
from typing import DefaultDict, List, Optional, Sequence, Tuple, TypeVar
import langsmith.beta._utils as beta_utils
import langsmith.schemas as ls_schemas
from langsmith import evaluation as ls_eval
from langsmith.client import Client
@beta_utils.warn_beta
def compute_test_metrics(project_name: str, *, evaluators: list, max_concurrency: Optional[int]=10, client: Optional[Client]=None) -> None:
    """Compute test metrics for a given test name using a list of evaluators.

    Args:
        project_name (str): The name of the test project to evaluate.
        evaluators (list): A list of evaluators to compute metrics with.
        max_concurrency (Optional[int], optional): The maximum number of concurrent
            evaluations. Defaults to 10.
        client (Optional[Client], optional): The client to use for evaluations.
            Defaults to None.

    Returns:
        None: This function does not return any value.
    """
    evaluators_: List[ls_eval.RunEvaluator] = []
    for func in evaluators:
        if isinstance(func, ls_eval.RunEvaluator):
            evaluators_.append(func)
        elif callable(func):
            evaluators_.append(ls_eval.run_evaluator(func))
        else:
            raise NotImplementedError(f'Evaluation not yet implemented for evaluator of type {type(func)}')
    client = client or Client()
    traces = _load_nested_traces(project_name, client)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        results = executor.map(client.evaluate_run, *zip(*_outer_product(traces, evaluators_)))
    for _ in results:
        pass