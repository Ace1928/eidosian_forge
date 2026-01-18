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
def add_model_construct_method(self, fields: list[PydanticModelField], config: ModelConfigData, is_settings: bool) -> None:
    """Adds a fully typed `model_construct` classmethod to the class.

        Similar to the fields-aware __init__ method, but always uses the field names (not aliases),
        and does not treat settings fields as optional.
        """
    set_str = self._api.named_type(f'{BUILTINS_NAME}.set', [self._api.named_type(f'{BUILTINS_NAME}.str')])
    optional_set_str = UnionType([set_str, NoneType()])
    fields_set_argument = Argument(Var('_fields_set', optional_set_str), optional_set_str, None, ARG_OPT)
    with state.strict_optional_set(self._api.options.strict_optional):
        args = self.get_field_arguments(fields, typed=True, requires_dynamic_aliases=False, use_alias=False, is_settings=is_settings)
    if not self.should_init_forbid_extra(fields, config):
        var = Var('kwargs')
        args.append(Argument(var, AnyType(TypeOfAny.explicit), None, ARG_STAR2))
    args = [fields_set_argument] + args
    add_method(self._api, self._cls, 'model_construct', args=args, return_type=fill_typevars(self._cls.info), is_classmethod=True)