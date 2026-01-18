from typing import Any, Dict, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def issue_create(self, query: str) -> str:
    try:
        import json
    except ImportError:
        raise ImportError('json is not installed. Please install it with `pip install json`')
    params = json.loads(query)
    return self.jira.issue_create(fields=dict(params))