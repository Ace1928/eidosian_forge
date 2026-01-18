from __future__ import annotations
import warnings
from abc import ABC
from string import Formatter
from typing import Any, Callable, Dict, List, Set, Tuple, Type
import langchain_core.utils.mustache as mustache
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, create_model
from langchain_core.utils import get_colored_text
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env
def mustache_schema(template: str) -> Type[BaseModel]:
    """Get the variables from a mustache template."""
    fields = set()
    prefix: Tuple[str, ...] = ()
    for type, key in mustache.tokenize(template):
        if key == '.':
            continue
        if type == 'end':
            prefix = prefix[:-key.count('.')]
        elif type == 'section':
            prefix = prefix + tuple(key.split('.'))
        elif type == 'variable':
            fields.add(prefix + tuple(key.split('.')))
    defs: Defs = {}
    while fields:
        field = fields.pop()
        current = defs
        for part in field[:-1]:
            current = current.setdefault(part, {})
        current[field[-1]] = {}
    return _create_model_recursive('PromptInput', defs)