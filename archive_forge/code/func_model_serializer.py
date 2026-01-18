from __future__ import annotations
import dataclasses
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
def model_serializer(__f: Callable[..., Any] | None=None, *, mode: Literal['plain', 'wrap']='plain', when_used: Literal['always', 'unless-none', 'json', 'json-unless-none']='always', return_type: Any=PydanticUndefined) -> Callable[[Any], Any]:
    """Decorator that enables custom model serialization.

    This is useful when a model need to be serialized in a customized manner, allowing for flexibility beyond just specific fields.

    An example would be to serialize temperature to the same temperature scale, such as degrees Celsius.

    ```python
    from typing import Literal

    from pydantic import BaseModel, model_serializer

    class TemperatureModel(BaseModel):
        unit: Literal['C', 'F']
        value: int

        @model_serializer()
        def serialize_model(self):
            if self.unit == 'F':
                return {'unit': 'C', 'value': int((self.value - 32) / 1.8)}
            return {'unit': self.unit, 'value': self.value}

    temperature = TemperatureModel(unit='F', value=212)
    print(temperature.model_dump())
    #> {'unit': 'C', 'value': 100}
    ```

    See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.

    Args:
        __f: The function to be decorated.
        mode: The serialization mode.

            - `'plain'` means the function will be called instead of the default serialization logic
            - `'wrap'` means the function will be called with an argument to optionally call the default
                serialization logic.
        when_used: Determines when this serializer should be used.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.

    Returns:
        The decorator function.
    """

    def dec(f: Callable[..., Any]) -> _decorators.PydanticDescriptorProxy[Any]:
        dec_info = _decorators.ModelSerializerDecoratorInfo(mode=mode, return_type=return_type, when_used=when_used)
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    if __f is None:
        return dec
    else:
        return dec(__f)