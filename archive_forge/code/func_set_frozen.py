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
def set_frozen(self, fields: List['PydanticModelField'], frozen: bool) -> None:
    """
        Marks all fields as properties so that attempts to set them trigger mypy errors.

        This is the same approach used by the attrs and dataclasses plugins.
        """
    ctx = self._ctx
    info = ctx.cls.info
    for field in fields:
        sym_node = info.names.get(field.name)
        if sym_node is not None:
            var = sym_node.node
            if isinstance(var, Var):
                var.is_property = frozen
            elif isinstance(var, PlaceholderNode) and (not ctx.api.final_iteration):
                ctx.api.defer()
            else:
                try:
                    var_str = str(var)
                except TypeError:
                    var_str = repr(var)
                detail = f'sym_node.node: {var_str} (of type {var.__class__})'
                error_unexpected_behavior(detail, ctx.api, ctx.cls)
        else:
            var = field.to_var(info, use_alias=False)
            var.info = info
            var.is_property = frozen
            var._fullname = get_fullname(info) + '.' + get_name(var)
            info.names[get_name(var)] = SymbolTableNode(MDEF, var)