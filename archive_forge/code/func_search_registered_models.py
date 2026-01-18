from typing import Any, Dict, List, Optional
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.client import MlflowClient
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.logging_utils import eprint
def search_registered_models(max_results: Optional[int]=None, filter_string: Optional[str]=None, order_by: Optional[List[str]]=None) -> List[RegisteredModel]:
    """Search for registered models that satisfy the filter criteria.

    Args:
        filter_string: Filter query string (e.g., "name = 'a_model_name' and tag.key = 'value1'"),
            defaults to searching for all registered models. The following identifiers, comparators,
            and logical operators are supported.

            Identifiers
              - "name": registered model name.
              - "tags.<tag_key>": registered model tag. If "tag_key" contains spaces, it must be
                wrapped with backticks (e.g., "tags.`extra key`").

            Comparators
              - "=": Equal to.
              - "!=": Not equal to.
              - "LIKE": Case-sensitive pattern match.
              - "ILIKE": Case-insensitive pattern match.

            Logical operators
              - "AND": Combines two sub-queries and returns True if both of them are True.

        max_results: If passed, specifies the maximum number of models desired. If not
            passed, all models will be returned.
        order_by: List of column names with ASC|DESC annotation, to be used for ordering
            matching search results.

    Returns:
        A list of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
        that satisfy the search expressions.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow
        from sklearn.linear_model import LogisticRegression

        with mlflow.start_run():
            mlflow.sklearn.log_model(
                LogisticRegression(),
                "Cordoba",
                registered_model_name="CordobaWeatherForecastModel",
            )
            mlflow.sklearn.log_model(
                LogisticRegression(),
                "Boston",
                registered_model_name="BostonWeatherForecastModel",
            )

        # Get search results filtered by the registered model name
        filter_string = "name = 'CordobaWeatherForecastModel'"
        results = mlflow.search_registered_models(filter_string=filter_string)
        print("-" * 80)
        for res in results:
            for mv in res.latest_versions:
                print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

        # Get search results filtered by the registered model name that matches
        # prefix pattern
        filter_string = "name LIKE 'Boston%'"
        results = mlflow.search_registered_models(filter_string=filter_string)
        print("-" * 80)
        for res in results:
            for mv in res.latest_versions:
                print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

        # Get all registered models and order them by ascending order of the names
        results = mlflow.search_registered_models(order_by=["name ASC"])
        print("-" * 80)
        for res in results:
            for mv in res.latest_versions:
                print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

    .. code-block:: text
        :caption: Output

        --------------------------------------------------------------------------------
        name=CordobaWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
        --------------------------------------------------------------------------------
        name=BostonWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
        --------------------------------------------------------------------------------
        name=BostonWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
        name=CordobaWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
    """

    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().search_registered_models(max_results=number_to_get, filter_string=filter_string, order_by=order_by, page_token=next_page_token)
    return get_results_from_paginated_fn(pagination_wrapper_func, SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT, max_results)