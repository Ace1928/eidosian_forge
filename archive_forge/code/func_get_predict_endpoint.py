import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
def get_predict_endpoint(self, label: str=None, ver: Union[str, int]=None):
    if label:
        return self.get_label(label) + ':predict'
    if ver:
        return self.get_version(ver) + ':predict'
    return self.default_predict