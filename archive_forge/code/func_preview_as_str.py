import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests import Request, Session
def preview_as_str(self, *args, **kwargs) -> str:
    """Same as preview, but returns a stringified version of the JSON for
        insertting back into an LLM."""
    data = self.preview(*args, **kwargs)
    return json.dumps(data)