import os
from langchain_community.agent_toolkits import ZapierToolkit
from langchain_community.utilities.zapier import ZapierNLAWrapper
from typing import Any, Dict, Optional
from langchain_core._api import warn_deprecated
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.tools import BaseTool
from langchain_community.tools.zapier.prompt import BASE_ZAPIER_TOOL_PROMPT
from langchain_community.utilities.zapier import ZapierNLAWrapper
@root_validator
def set_name_description(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    zapier_description = values['zapier_description']
    params_schema = values['params_schema']
    if 'instructions' in params_schema:
        del params_schema['instructions']
    necessary_fields = {'{zapier_description}', '{params}'}
    if not all((field in values['base_prompt'] for field in necessary_fields)):
        raise ValueError('Your custom base Zapier prompt must contain input fields for {zapier_description} and {params}.')
    values['name'] = zapier_description
    values['description'] = values['base_prompt'].format(zapier_description=zapier_description, params=str(list(params_schema.keys())))
    return values