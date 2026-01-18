from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def no_info_after_validator_function(function: NoInfoValidatorFunction, schema: CoreSchema, *, ref: str | None=None, metadata: Any=None, serialization: SerSchema | None=None) -> AfterValidatorFunctionSchema:
    """
    Returns a schema that calls a validator function after validating, no `info` argument is provided, e.g.:

    ```py
    from pydantic_core import SchemaValidator, core_schema

    def fn(v: str) -> str:
        return v + 'world'

    func_schema = core_schema.no_info_after_validator_function(fn, core_schema.str_schema())
    schema = core_schema.typed_dict_schema({'a': core_schema.typed_dict_field(func_schema)})

    v = SchemaValidator(schema)
    assert v.validate_python({'a': b'hello '}) == {'a': 'hello world'}
    ```

    Args:
        function: The validator function to call after the schema is validated
        schema: The schema to validate before the validator function
        ref: optional unique identifier of the schema, used to reference the schema in other places
        metadata: Any other information you want to include with the schema, not used by pydantic-core
        serialization: Custom serialization schema
    """
    return _dict_not_none(type='function-after', function={'type': 'no-info', 'function': function}, schema=schema, ref=ref, metadata=metadata, serialization=serialization)