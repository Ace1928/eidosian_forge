from __future__ import annotations as _annotations
from inspect import Parameter, signature
from typing import Any, Dict, Tuple, Union, cast
from pydantic_core import core_schema
from typing_extensions import Protocol
from ..errors import PydanticUserError
from ._decorators import can_be_positional
def can_be_keyword(param: Parameter) -> bool:
    return param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)