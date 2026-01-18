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
def get_template_variables(template: str, template_format: str) -> List[str]:
    """Get the variables from the template.

    Args:
        template: The template string.
        template_format: The template format. Should be one of "f-string" or "jinja2".

    Returns:
        The variables from the template.

    Raises:
        ValueError: If the template format is not supported.
    """
    if template_format == 'jinja2':
        input_variables = _get_jinja2_variables_from_template(template)
    elif template_format == 'f-string':
        input_variables = {v for _, v, _, _ in Formatter().parse(template) if v is not None}
    elif template_format == 'mustache':
        input_variables = mustache_template_vars(template)
    else:
        raise ValueError(f'Unsupported template format: {template_format}')
    return sorted(input_variables)