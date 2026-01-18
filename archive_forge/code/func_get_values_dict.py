from __future__ import annotations
import sys
from configparser import ConfigParser
from typing import Any, Callable, Iterator
from mypy.errorcodes import ErrorCode
from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.plugins.common import (
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.state import state
from mypy.typeops import map_type_from_supertype
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic._internal import _fields
from pydantic.version import parse_mypy_version
def get_values_dict(self) -> dict[str, Any]:
    """Returns a dict of Pydantic model config names to their values.

        It includes the config if config value is not `None`.
        """
    return {k: v for k, v in self.__dict__.items() if v is not None}