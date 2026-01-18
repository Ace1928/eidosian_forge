from __future__ import annotations
import json
import logging
import time
from typing import List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import validator
from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool
@validator('providers')
def check_only_one_provider_selected(cls, v: List[str]) -> List[str]:
    """
        This tool has no feature to combine providers results.
        Therefore we only allow one provider
        """
    if len(v) > 1:
        raise ValueError('Please select only one provider. The feature to combine providers results is not available for this tool.')
    return v