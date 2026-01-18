from __future__ import annotations
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import (
import docstring_parser
import typing_extensions
from typing_extensions import (
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def field_list_from_callable(f: Union[Callable, TypeForm[Any]], default_instance: DefaultInstance, support_single_arg_types: bool) -> Tuple[Union[Callable, TypeForm[Any]], Dict[TypeVar, TypeForm], List[FieldDefinition]]:
    """Generate a list of generic 'field' objects corresponding to the inputs of some
    annotated callable.

    Returns:
        The type that `f` is resolved as.
        A type_from_typevar dict.
        A list of field definitions.
    """
    f, type_from_typevar = _resolver.resolve_generic_types(f)
    f = _resolver.unwrap_newtype_and_narrow_subtypes(f, default_instance)
    field_list = _try_field_list_from_callable(f, default_instance)
    if isinstance(field_list, UnsupportedNestedTypeMessage):
        if support_single_arg_types:
            return (f, type_from_typevar, [FieldDefinition(intern_name='value', extern_name='value', type_or_callable=f, default=default_instance, is_default_from_default_instance=True, helptext='', custom_constructor=False, markers=frozenset((_markers.Positional, _markers._PositionalCall)), argconf=_confstruct._ArgConfiguration(None, None, None, None, None, None), call_argname='')])
        else:
            raise _instantiators.UnsupportedTypeAnnotationError(field_list.message)
    _, parent_markers = _resolver.unwrap_annotated(f, _markers._Marker)
    field_list = list(map(lambda field: field.add_markers(parent_markers), field_list))

    def resolve(field: FieldDefinition) -> FieldDefinition:
        typ = field.type_or_callable
        typ = _resolver.apply_type_from_typevar(typ, type_from_typevar)
        typ = _resolver.type_from_typevar_constraints(typ)
        typ = _resolver.narrow_collection_types(typ, field.default)
        typ = _resolver.narrow_union_type(typ, field.default)
        if type(typ) is type and (not isinstance(field.default, typ)) and (not field.custom_constructor) and (field.default not in DEFAULT_SENTINEL_SINGLETONS) and (not isinstance(field.default, numbers.Number)):
            warnings.warn(f"The field {field.intern_name} is annotated with type {field.type_or_callable}, but the default value {field.default} has type {type(field.default)}. We'll try to handle this gracefully, but it may cause unexpected behavior.")
            typ = Union[typ, type(field.default)]
        field = dataclasses.replace(field, type_or_callable=typ)
        return field
    field_list = list(map(resolve, field_list))
    return (f, type_from_typevar, field_list)