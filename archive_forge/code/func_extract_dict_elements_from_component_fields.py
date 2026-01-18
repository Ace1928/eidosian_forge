import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def extract_dict_elements_from_component_fields(data: dict, component: Type[Component]) -> dict:
    """Extract elements from a dictionary.

    Args:
        data: The dictionary to extract elements from.
        component: The component to extract elements from.

    Returns:
        A dictionary containing the elements from the input dictionary that are also
        in the component.
    """
    output = {}
    for attribute in fields(component):
        if attribute.name in data:
            output[attribute.name] = data[attribute.name]
    return output