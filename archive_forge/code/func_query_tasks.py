import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def query_tasks(self, query: str) -> Dict:
    """
        Query tasks that match certain fields
        """
    params, error = load_query(query, fault_tolerant=True)
    if params is None:
        return {'Error': error}
    url = f'{DEFAULT_URL}/list/{params['list_id']}/task'
    params = self.get_default_params()
    response = requests.get(url, headers=self.get_headers(), params=params)
    return {'response': response}