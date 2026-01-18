from __future__ import annotations
import logging
from typing import Dict, List, Literal, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field, root_validator, validator
from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool
@root_validator
def check_voice_models_key_is_provider_name(cls, values: dict) -> dict:
    for key in values.get('voice_models', {}).keys():
        if key not in values.get('providers', []):
            raise ValueError('voice_model should be formatted like this {<provider_name>: <its_voice_model>}')
    return values