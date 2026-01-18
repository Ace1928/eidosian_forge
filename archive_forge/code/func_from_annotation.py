from __future__ import annotations as _annotations
import dataclasses
import inspect
import typing
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Any, ClassVar
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, Unpack
from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .warnings import PydanticDeprecatedSince20
@staticmethod
def from_annotation(annotation: type[Any]) -> FieldInfo:
    """Creates a `FieldInfo` instance from a bare annotation.

        This function is used internally to create a `FieldInfo` from a bare annotation like this:

        ```python
        import pydantic

        class MyModel(pydantic.BaseModel):
            foo: int  # <-- like this
        ```

        We also account for the case where the annotation can be an instance of `Annotated` and where
        one of the (not first) arguments in `Annotated` is an instance of `FieldInfo`, e.g.:

        ```python
        import annotated_types
        from typing_extensions import Annotated

        import pydantic

        class MyModel(pydantic.BaseModel):
            foo: Annotated[int, annotated_types.Gt(42)]
            bar: Annotated[int, pydantic.Field(gt=42)]
        ```

        Args:
            annotation: An annotation object.

        Returns:
            An instance of the field metadata.
        """
    final = False
    if _typing_extra.is_finalvar(annotation):
        final = True
        if annotation is not typing_extensions.Final:
            annotation = typing_extensions.get_args(annotation)[0]
    if _typing_extra.is_annotated(annotation):
        first_arg, *extra_args = typing_extensions.get_args(annotation)
        if _typing_extra.is_finalvar(first_arg):
            final = True
        field_info_annotations = [a for a in extra_args if isinstance(a, FieldInfo)]
        field_info = FieldInfo.merge_field_infos(*field_info_annotations, annotation=first_arg)
        if field_info:
            new_field_info = copy(field_info)
            new_field_info.annotation = first_arg
            new_field_info.frozen = final or field_info.frozen
            metadata: list[Any] = []
            for a in extra_args:
                if not isinstance(a, FieldInfo):
                    metadata.append(a)
                else:
                    metadata.extend(a.metadata)
            new_field_info.metadata = metadata
            return new_field_info
    return FieldInfo(annotation=annotation, frozen=final or None)