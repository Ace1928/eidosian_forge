import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def get_lists(self) -> Dict:
    """
        Get all available lists.
        """
    url = f'{DEFAULT_URL}/folder/{self.folder_id}/list'
    params = self.get_default_params()
    response = requests.get(url, headers=self.get_headers(), params=params)
    return {'response': response}