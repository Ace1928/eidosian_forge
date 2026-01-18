import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
@validator('cluster_id', always=True)
def set_cluster_id(cls, v: Any, values: Dict[str, Any]) -> Optional[str]:
    if v and values['endpoint_name']:
        raise ValueError('Cannot set both endpoint_name and cluster_id.')
    elif values['endpoint_name']:
        return None
    elif v:
        return v
    else:
        try:
            if (v := get_repl_context().clusterId):
                return v
            raise ValueError("Context doesn't contain clusterId.")
        except Exception as e:
            raise ValueError(f'Neither endpoint_name nor cluster_id was set. And the cluster_id cannot be automatically determined. Received error: {e}')