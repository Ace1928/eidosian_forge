from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, final
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def is_valid_now(self) -> bool:
    return self._access_token is not None and self._customer_id is not None and (self._access_token_expiry is not None) and (self._access_token_expiry > datetime.now())