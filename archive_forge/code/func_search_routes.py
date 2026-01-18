from typing import Any, Dict, List, Optional
from mlflow.gateway.client import MlflowGatewayClient
from mlflow.gateway.config import LimitsConfig, Route
from mlflow.gateway.constants import MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
from mlflow.gateway.utils import gateway_deprecated
from mlflow.utils import get_results_from_paginated_fn
@gateway_deprecated
def search_routes() -> List[Route]:
    """
    Searches for routes in the MLflow Gateway service.

    This function creates an instance of MlflowGatewayClient and uses it to fetch a list of routes
    from the Gateway service.

    Returns:
        A list of Route instances.

    """

    def pagination_wrapper_func(_, next_page_token):
        return MlflowGatewayClient().search_routes(page_token=next_page_token)
    return get_results_from_paginated_fn(paginated_fn=pagination_wrapper_func, max_results_per_page=MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE, max_results=None)