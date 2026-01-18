import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def get_headers(self) -> Mapping[str, Union[str, bytes]]:
    """Get the headers for the request."""
    if not isinstance(self.access_token, str):
        raise TypeError(f'Access Token: {self.access_token}, must be str.')
    headers = {'Authorization': str(self.access_token), 'Content-Type': 'application/json'}
    return headers