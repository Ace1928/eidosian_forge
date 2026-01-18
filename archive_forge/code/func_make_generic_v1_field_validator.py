from __future__ import annotations as _annotations
from inspect import Parameter, signature
from typing import Any, Dict, Tuple, Union, cast
from pydantic_core import core_schema
from typing_extensions import Protocol
from ..errors import PydanticUserError
from ._decorators import can_be_positional
def make_generic_v1_field_validator(validator: V1Validator) -> core_schema.WithInfoValidatorFunction:
    """Wrap a V1 style field validator for V2 compatibility.

    Args:
        validator: The V1 style field validator.

    Returns:
        A wrapped V2 style field validator.

    Raises:
        PydanticUserError: If the signature is not supported or the parameters are
            not available in Pydantic V2.
    """
    sig = signature(validator)
    needs_values_kw = False
    for param_num, (param_name, parameter) in enumerate(sig.parameters.items()):
        if can_be_keyword(parameter) and param_name in ('field', 'config'):
            raise PydanticUserError('The `field` and `config` parameters are not available in Pydantic V2, please use the `info` parameter instead.', code='validator-field-config-info')
        if parameter.kind is Parameter.VAR_KEYWORD:
            needs_values_kw = True
        elif can_be_keyword(parameter) and param_name == 'values':
            needs_values_kw = True
        elif can_be_positional(parameter) and param_num == 0:
            continue
        elif parameter.default is Parameter.empty:
            raise PydanticUserError(f'Unsupported signature for V1 style validator {validator}: {sig} is not supported.', code='validator-v1-signature')
    if needs_values_kw:
        val1 = cast(V1ValidatorWithValues, validator)

        def wrapper1(value: Any, info: core_schema.ValidationInfo) -> Any:
            return val1(value, values=info.data)
        return wrapper1
    else:
        val2 = cast(V1OnlyValueValidator, validator)

        def wrapper2(value: Any, _: core_schema.ValidationInfo) -> Any:
            return val2(value)
        return wrapper2