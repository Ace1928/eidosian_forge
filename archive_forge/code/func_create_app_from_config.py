from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from mlflow.deployments.server.config import Endpoint
from mlflow.deployments.server.constants import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import SetLimitsModel
from mlflow.gateway.config import (
from mlflow.gateway.constants import (
from mlflow.gateway.providers import get_provider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import SearchRoutesToken, make_streaming_response
from mlflow.version import VERSION
def create_app_from_config(config: GatewayConfig) -> GatewayAPI:
    """
    Create the GatewayAPI app from the gateway configuration.
    """
    limiter = Limiter(key_func=get_remote_address, storage_uri=MLFLOW_GATEWAY_RATE_LIMITS_STORAGE_URI.get())
    app = GatewayAPI(config=config, limiter=limiter, title='MLflow Deployments Server', description='The core deployments API for reverse proxy interface using remote inference endpoints within MLflow', version=VERSION, docs_url=None)

    @app.get('/', include_in_schema=False)
    async def index():
        return RedirectResponse(url='/docs')

    @app.get('/favicon.ico', include_in_schema=False)
    async def favicon():
        for directory in ['build', 'public']:
            favicon = Path(__file__).parent.parent.parent.joinpath('server', 'js', directory, 'favicon.ico')
            if favicon.exists():
                return FileResponse(favicon)
        raise HTTPException(status_code=404, detail='favicon.ico not found')

    @app.get('/docs', include_in_schema=False)
    async def docs():
        return get_swagger_ui_html(openapi_url='/openapi.json', title='MLflow Deployments Server', swagger_favicon_url='/favicon.ico')

    @app.get(MLFLOW_DEPLOYMENTS_HEALTH_ENDPOINT)
    @app.get(MLFLOW_GATEWAY_HEALTH_ENDPOINT, include_in_schema=False)
    async def health() -> HealthResponse:
        return {'status': 'OK'}

    @app.get(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE + '{endpoint_name}')
    async def get_endpoint(endpoint_name: str) -> Endpoint:
        if (matched := app.get_dynamic_route(endpoint_name)):
            return matched.to_endpoint()
        raise HTTPException(status_code=404, detail=f"The endpoint '{endpoint_name}' is not present or active on the server. Please verify the endpoint name.")

    @app.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE + '{route_name}', include_in_schema=False)
    async def get_route(route_name: str) -> Route:
        if (matched := app.get_dynamic_route(route_name)):
            return matched
        raise HTTPException(status_code=404, detail=f"The route '{route_name}' is not present or active on the server. Please verify the route name.")

    @app.get(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE)
    async def list_endpoints(page_token: Optional[str]=None) -> ListEndpointsResponse:
        start_idx = SearchRoutesToken.decode(page_token).index if page_token is not None else 0
        end_idx = start_idx + MLFLOW_DEPLOYMENTS_LIST_ENDPOINTS_PAGE_SIZE
        routes = list(app.dynamic_routes.values())
        result = {'endpoints': [route.to_endpoint() for route in routes[start_idx:end_idx]]}
        if len(routes[end_idx:]) > 0:
            next_page_token = SearchRoutesToken(index=end_idx)
            result['next_page_token'] = next_page_token.encode()
        return result

    @app.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE, include_in_schema=False)
    async def search_routes(page_token: Optional[str]=None) -> SearchRoutesResponse:
        start_idx = SearchRoutesToken.decode(page_token).index if page_token is not None else 0
        end_idx = start_idx + MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
        routes = list(app.dynamic_routes.values())
        result = {'routes': routes[start_idx:end_idx]}
        if len(routes[end_idx:]) > 0:
            next_page_token = SearchRoutesToken(index=end_idx)
            result['next_page_token'] = next_page_token.encode()
        return result

    @app.get(MLFLOW_DEPLOYMENTS_LIMITS_BASE + '{endpoint}')
    @app.get(MLFLOW_GATEWAY_LIMITS_BASE + '{endpoint}', include_in_schema=False)
    async def get_limits(endpoint: str) -> LimitsConfig:
        raise HTTPException(status_code=501, detail='The get_limits API is not available yet.')

    @app.post(MLFLOW_DEPLOYMENTS_LIMITS_BASE)
    @app.post(MLFLOW_GATEWAY_LIMITS_BASE, include_in_schema=False)
    async def set_limits(payload: SetLimitsModel) -> LimitsConfig:
        raise HTTPException(status_code=501, detail='The set_limits API is not available yet.')
    return app