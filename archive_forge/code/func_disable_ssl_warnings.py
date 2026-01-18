import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import (
from langchain_core.utils import get_from_dict_or_env
@validator('unsecure')
def disable_ssl_warnings(cls, v: bool) -> bool:
    """Disable SSL warnings."""
    if v:
        try:
            import urllib3
            urllib3.disable_warnings()
        except ImportError as e:
            print(e)
    return v