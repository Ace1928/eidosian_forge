from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Generic, TypeVar
from pydantic_core import SchemaSerializer, SchemaValidator
from typing_extensions import Literal
from ..errors import PydanticErrorCodes, PydanticUserError
def set_model_mocks(cls: type[BaseModel], cls_name: str, undefined_name: str='all referenced types') -> None:
    """Set `__pydantic_validator__` and `__pydantic_serializer__` to `MockValSer`s on a model.

    Args:
        cls: The model class to set the mocks on
        cls_name: Name of the model class, used in error messages
        undefined_name: Name of the undefined thing, used in error messages
    """
    undefined_type_error_message = f'`{cls_name}` is not fully defined; you should define {undefined_name}, then call `{cls_name}.model_rebuild()`.'

    def attempt_rebuild_validator() -> SchemaValidator | None:
        if cls.model_rebuild(raise_errors=False, _parent_namespace_depth=5) is not False:
            return cls.__pydantic_validator__
        else:
            return None
    cls.__pydantic_validator__ = MockValSer(undefined_type_error_message, code='class-not-fully-defined', val_or_ser='validator', attempt_rebuild=attempt_rebuild_validator)

    def attempt_rebuild_serializer() -> SchemaSerializer | None:
        if cls.model_rebuild(raise_errors=False, _parent_namespace_depth=5) is not False:
            return cls.__pydantic_serializer__
        else:
            return None
    cls.__pydantic_serializer__ = MockValSer(undefined_type_error_message, code='class-not-fully-defined', val_or_ser='serializer', attempt_rebuild=attempt_rebuild_serializer)