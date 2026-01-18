import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from .inference._client import InferenceClient
from .inference._generated._async_client import AsyncInferenceClient
from .utils import logging, parse_datetime
@classmethod
def from_raw(cls, raw: Dict, namespace: str, token: Optional[str]=None, api: Optional['HfApi']=None) -> 'InferenceEndpoint':
    """Initialize object from raw dictionary."""
    if api is None:
        from .hf_api import HfApi
        api = HfApi()
    if token is None:
        token = api.token
    return cls(raw=raw, namespace=namespace, _token=token, _api=api)