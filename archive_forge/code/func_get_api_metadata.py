import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
@timed_cache(seconds=600)
def get_api_metadata(self):
    return {model: self.endpoints[model].get_metadata() for model in self.available_models}