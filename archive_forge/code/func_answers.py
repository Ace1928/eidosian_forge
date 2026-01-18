import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import (
from langchain_core.utils import get_from_dict_or_env
@property
def answers(self) -> Any:
    """Helper accessor on the json result."""
    return self.get('answers')