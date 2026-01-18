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
def add_initializer(self, fields: List['PydanticModelField'], config: 'ModelConfigData', is_settings: bool) -> None:
    """
        Adds a fields-aware `__init__` method to the class.

        The added `__init__` will be annotated with types vs. all `Any` depending on the plugin settings.
        """
    ctx = self._ctx
    typed = self.plugin_config.init_typed
    use_alias = config.allow_population_by_field_name is not True
    force_all_optional = is_settings or bool(config.has_alias_generator and (not config.allow_population_by_field_name))
    init_arguments = self.get_field_arguments(fields, typed=typed, force_all_optional=force_all_optional, use_alias=use_alias)
    if not self.should_init_forbid_extra(fields, config):
        var = Var('kwargs')
        init_arguments.append(Argument(var, AnyType(TypeOfAny.explicit), None, ARG_STAR2))
    if '__init__' not in ctx.cls.info.names:
        add_method(ctx, '__init__', init_arguments, NoneType())