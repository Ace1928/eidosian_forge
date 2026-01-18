import logging
from typing import Any, Dict, Optional
from ray import cloudpickle
from ray.serve._private.common import EndpointInfo, EndpointTag
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
def get_endpoints(self) -> Dict[EndpointTag, Dict[str, Any]]:
    endpoints = {}
    for endpoint, info in self._endpoints.items():
        endpoints[endpoint] = {'route': info.route}
    return endpoints