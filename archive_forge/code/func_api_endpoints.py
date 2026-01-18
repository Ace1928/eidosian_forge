import json
import functools
from lazyops.types.models import BaseModel, validator
from lazyops.types.classprops import lazyproperty
from lazyops.types.static import RESPONSE_SUCCESS_CODES
from lazyops.types.resources import BaseResource, ResourceType, ResponseResource, ResponseResourceType
from lazyops.types.errors import ClientError, fatal_exception
from lazyops.imports._aiohttpx import aiohttpx, resolve_aiohttpx
from lazyops.imports._backoff import backoff, require_backoff
from lazyops.configs.base import DefaultSettings
from lazyops.utils.logs import default_logger as logger
from lazyops.utils.serialization import ObjectEncoder
from typing import Optional, Dict, List, Any, Type, Callable
@lazyproperty
def api_endpoints(self) -> Dict[str, Dict[str, str]]:
    """
        Returns the api endpoints and methods
        """
    return {'get': {'url': '/v1/get', 'method': 'GET'}, 'list': {'url': '/v1/list', 'method': 'GET'}, 'create': {'url': '/v1/create', 'method': 'POST'}, 'update': {'url': '/v1/update', 'method': 'PUT'}, 'delete': {'url': '/v1/delete', 'method': 'DELETE'}}