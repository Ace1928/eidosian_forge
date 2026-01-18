from typing import Any, Dict, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def parse_projects(self, projects: List[dict]) -> List[dict]:
    parsed = []
    for project in projects:
        id = project['id']
        key = project['key']
        name = project['name']
        type = project['projectTypeKey']
        style = project['style']
        parsed.append({'id': id, 'key': key, 'name': name, 'type': type, 'style': style})
    return parsed