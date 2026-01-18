from __future__ import annotations
import functools
from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeVar
from pydantic_core import CoreConfig, CoreSchema, SchemaValidator, ValidationError
from typing_extensions import Literal, ParamSpec
def create_schema_validator(schema: CoreSchema, schema_type: Any, schema_type_module: str, schema_type_name: str, schema_kind: SchemaKind, config: CoreConfig | None=None, plugin_settings: dict[str, Any] | None=None) -> SchemaValidator:
    """Create a `SchemaValidator` or `PluggableSchemaValidator` if plugins are installed.

    Returns:
        If plugins are installed then return `PluggableSchemaValidator`, otherwise return `SchemaValidator`.
    """
    from . import SchemaTypePath
    from ._loader import get_plugins
    plugins = get_plugins()
    if plugins:
        return PluggableSchemaValidator(schema, schema_type, SchemaTypePath(schema_type_module, schema_type_name), schema_kind, config, plugins, plugin_settings or {})
    else:
        return SchemaValidator(schema, config)