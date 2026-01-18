import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def create_list(self, query: str) -> Dict:
    """
        Creates a new list.
        """
    query_dict, error = load_query(query, fault_tolerant=True)
    if query_dict is None:
        return {'Error': error}
    location = self.folder_id if self.folder_id else self.space_id
    url = f'{DEFAULT_URL}/folder/{location}/list'
    payload = extract_dict_elements_from_component_fields(query_dict, Task)
    headers = self.get_headers()
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    parsed_list = parse_dict_through_component(data, CUList, fault_tolerant=True)
    if 'id' in parsed_list:
        self.list_id = parsed_list['id']
    return parsed_list