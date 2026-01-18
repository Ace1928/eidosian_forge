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
def get_class_decorator_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
    """Mark pydantic.dataclasses as dataclass.

        Mypy version 1.1.1 added support for `@dataclass_transform` decorator.
        """
    if fullname == DATACLASS_FULLNAME and MYPY_VERSION_TUPLE < (1, 1):
        return dataclasses.dataclass_class_maker_callback
    return None