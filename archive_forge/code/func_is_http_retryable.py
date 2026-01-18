from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, final
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def is_http_retryable(rsp: requests.Response) -> bool:
    return bool(rsp) and rsp.status_code in [408, 425, 429, 500, 502, 503, 504]