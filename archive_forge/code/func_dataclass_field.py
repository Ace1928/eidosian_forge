from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def dataclass_field(name: str, schema: CoreSchema, *, kw_only: bool | None=None, init: bool | None=None, init_only: bool | None=None, validation_alias: str | list[str | int] | list[list[str | int]] | None=None, serialization_alias: str | None=None, serialization_exclude: bool | None=None, metadata: Any=None, frozen: bool | None=None) -> DataclassField:
    """
    Returns a schema for a dataclass field, e.g.:

    ```py
    from pydantic_core import SchemaValidator, core_schema

    field = core_schema.dataclass_field(
        name='a', schema=core_schema.str_schema(), kw_only=False
    )
    schema = core_schema.dataclass_args_schema('Foobar', [field])
    v = SchemaValidator(schema)
    assert v.validate_python({'a': 'hello'}) == ({'a': 'hello'}, None)
    ```

    Args:
        name: The name to use for the argument parameter
        schema: The schema to use for the argument parameter
        kw_only: Whether the field can be set with a positional argument as well as a keyword argument
        init: Whether the field should be validated during initialization
        init_only: Whether the field should be omitted  from `__dict__` and passed to `__post_init__`
        validation_alias: The alias(es) to use to find the field in the validation data
        serialization_alias: The alias to use as a key when serializing
        serialization_exclude: Whether to exclude the field when serializing
        metadata: Any other information you want to include with the schema, not used by pydantic-core
        frozen: Whether the field is frozen
    """
    return _dict_not_none(type='dataclass-field', name=name, schema=schema, kw_only=kw_only, init=init, init_only=init_only, validation_alias=validation_alias, serialization_alias=serialization_alias, serialization_exclude=serialization_exclude, metadata=metadata, frozen=frozen)