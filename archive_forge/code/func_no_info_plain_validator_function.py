from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def no_info_plain_validator_function(function: NoInfoValidatorFunction, *, ref: str | None=None, metadata: Any=None, serialization: SerSchema | None=None) -> PlainValidatorFunctionSchema:
    """
    Returns a schema that uses the provided function for validation, no `info` argument is passed, e.g.:

    ```py
    from pydantic_core import SchemaValidator, core_schema

    def fn(v: str) -> str:
        assert 'hello' in v
        return v + 'world'

    schema = core_schema.no_info_plain_validator_function(function=fn)
    v = SchemaValidator(schema)
    assert v.validate_python('hello ') == 'hello world'
    ```

    Args:
        function: The validator function to call
        ref: optional unique identifier of the schema, used to reference the schema in other places
        metadata: Any other information you want to include with the schema, not used by pydantic-core
        serialization: Custom serialization schema
    """
    return _dict_not_none(type='function-plain', function={'type': 'no-info', 'function': function}, ref=ref, metadata=metadata, serialization=serialization)