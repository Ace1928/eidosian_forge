import logging
from typing import Any, Dict, Optional
from ray import cloudpickle
from ray.serve._private.common import EndpointInfo, EndpointTag
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
def delete_endpoint(self, endpoint: EndpointTag) -> None:
    if endpoint not in self._endpoints:
        return
    del self._endpoints[endpoint]
    self._checkpoint()
    self._notify_route_table_changed()