from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Any, Type, Union, Generic, TypeVar, Callable, cast
from datetime import date, datetime
from typing_extensions import (
import pydantic
import pydantic.generics
from pydantic.fields import FieldInfo
from ._types import (
from ._utils import (
from ._compat import (
from ._constants import RAW_RESPONSE_HEADER
@override
def model_dump_json(self, *, indent: int | None=None, include: IncEx=None, exclude: IncEx=None, by_alias: bool=False, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, round_trip: bool=False, warnings: bool=True) -> str:
    """Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump_json

            Generates a JSON representation of the model using Pydantic's `to_json` method.

            Args:
                indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
                include: Field(s) to include in the JSON output. Can take either a string or set of strings.
                exclude: Field(s) to exclude from the JSON output. Can take either a string or set of strings.
                by_alias: Whether to serialize using field aliases.
                exclude_unset: Whether to exclude fields that have not been explicitly set.
                exclude_defaults: Whether to exclude fields that have the default value.
                exclude_none: Whether to exclude fields that have a value of `None`.
                round_trip: Whether to use serialization/deserialization between JSON and class instance.
                warnings: Whether to show any warnings that occurred during serialization.

            Returns:
                A JSON string representation of the model.
            """
    if round_trip != False:
        raise ValueError('round_trip is only supported in Pydantic v2')
    if warnings != True:
        raise ValueError('warnings is only supported in Pydantic v2')
    return super().json(indent=indent, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)