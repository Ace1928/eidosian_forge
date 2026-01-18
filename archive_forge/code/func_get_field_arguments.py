import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name  # type: ignore
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.utils import is_valid_field
def get_field_arguments(self, fields: List['PydanticModelField'], typed: bool, force_all_optional: bool, use_alias: bool) -> List[Argument]:
    """
        Helper function used during the construction of the `__init__` and `construct` method signatures.

        Returns a list of mypy Argument instances for use in the generated signatures.
        """
    info = self._ctx.cls.info
    arguments = [field.to_argument(info, typed=typed, force_optional=force_all_optional, use_alias=use_alias) for field in fields if not (use_alias and field.has_dynamic_alias)]
    return arguments