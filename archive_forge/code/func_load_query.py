import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def load_query(query: str, fault_tolerant: bool=False) -> Tuple[Optional[Dict], Optional[str]]:
    """Attempts to parse a JSON string and return the parsed object.

    If parsing fails, returns an error message.

    :param query: The JSON string to parse.
    :return: A tuple containing the parsed object or None and an error message or None.
    """
    try:
        return (json.loads(query), None)
    except json.JSONDecodeError as e:
        if fault_tolerant:
            return (None, f'Input must be a valid JSON. Got the following error: {str(e)}. \n"Please reformat and try again.')
        else:
            raise e