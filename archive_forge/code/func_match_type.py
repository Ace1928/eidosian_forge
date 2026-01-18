from __future__ import annotations as _annotations
import collections.abc
import dataclasses
import inspect
import re
import sys
import typing
import warnings
from contextlib import contextmanager
from copy import copy, deepcopy
from enum import Enum
from functools import partial
from inspect import Parameter, _ParameterKind, signature
from itertools import chain
from operator import attrgetter
from types import FunctionType, LambdaType, MethodType
from typing import (
from warnings import warn
from pydantic_core import CoreSchema, PydanticUndefined, core_schema, to_jsonable_python
from typing_extensions import Annotated, Literal, TypeAliasType, TypedDict, get_args, get_origin, is_typeddict
from ..aliases import AliasGenerator
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from ..config import ConfigDict, JsonDict, JsonEncoder
from ..errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation, PydanticUserError
from ..json_schema import JsonSchemaValue
from ..version import version_short
from ..warnings import PydanticDeprecatedSince20
from . import _core_utils, _decorators, _discriminated_union, _known_annotated_metadata, _typing_extra
from ._config import ConfigWrapper, ConfigWrapperStack
from ._core_metadata import CoreMetadataHandler, build_metadata_dict
from ._core_utils import (
from ._decorators import (
from ._fields import collect_dataclass_fields, get_type_hints_infer_globalns
from ._forward_ref import PydanticRecursiveRef
from ._generics import get_standard_typevars_map, has_instance_in_type, recursively_defined_type_refs, replace_types
from ._schema_generation_shared import (
from ._typing_extra import is_finalvar
from ._utils import lenient_issubclass
def match_type(self, obj: Any) -> core_schema.CoreSchema:
    """Main mapping of types to schemas.

        The general structure is a series of if statements starting with the simple cases
        (non-generic primitive types) and then handling generics and other more complex cases.

        Each case either generates a schema directly, calls into a public user-overridable method
        (like `GenerateSchema.tuple_variable_schema`) or calls into a private method that handles some
        boilerplate before calling into the user-facing method (e.g. `GenerateSchema._tuple_schema`).

        The idea is that we'll evolve this into adding more and more user facing methods over time
        as they get requested and we figure out what the right API for them is.
        """
    if obj is str:
        return self.str_schema()
    elif obj is bytes:
        return core_schema.bytes_schema()
    elif obj is int:
        return core_schema.int_schema()
    elif obj is float:
        return core_schema.float_schema()
    elif obj is bool:
        return core_schema.bool_schema()
    elif obj is Any or obj is object:
        return core_schema.any_schema()
    elif obj is None or obj is _typing_extra.NoneType:
        return core_schema.none_schema()
    elif obj in TUPLE_TYPES:
        return self._tuple_schema(obj)
    elif obj in LIST_TYPES:
        return self._list_schema(obj, self._get_first_arg_or_any(obj))
    elif obj in SET_TYPES:
        return self._set_schema(obj, self._get_first_arg_or_any(obj))
    elif obj in FROZEN_SET_TYPES:
        return self._frozenset_schema(obj, self._get_first_arg_or_any(obj))
    elif obj in DICT_TYPES:
        return self._dict_schema(obj, *self._get_first_two_args_or_any(obj))
    elif isinstance(obj, TypeAliasType):
        return self._type_alias_type_schema(obj)
    elif obj == type:
        return self._type_schema()
    elif _typing_extra.is_callable_type(obj):
        return core_schema.callable_schema()
    elif _typing_extra.is_literal_type(obj):
        return self._literal_schema(obj)
    elif is_typeddict(obj):
        return self._typed_dict_schema(obj, None)
    elif _typing_extra.is_namedtuple(obj):
        return self._namedtuple_schema(obj, None)
    elif _typing_extra.is_new_type(obj):
        return self.generate_schema(obj.__supertype__)
    elif obj == re.Pattern:
        return self._pattern_schema(obj)
    elif obj is collections.abc.Hashable or obj is typing.Hashable:
        return self._hashable_schema()
    elif isinstance(obj, typing.TypeVar):
        return self._unsubstituted_typevar_schema(obj)
    elif is_finalvar(obj):
        if obj is Final:
            return core_schema.any_schema()
        return self.generate_schema(self._get_first_arg_or_any(obj))
    elif isinstance(obj, (FunctionType, LambdaType, MethodType, partial)):
        return self._callable_schema(obj)
    elif inspect.isclass(obj) and issubclass(obj, Enum):
        from ._std_types_schema import get_enum_core_schema
        return get_enum_core_schema(obj, self._config_wrapper.config_dict)
    if _typing_extra.is_dataclass(obj):
        return self._dataclass_schema(obj, None)
    res = self._get_prepare_pydantic_annotations_for_known_type(obj, ())
    if res is not None:
        source_type, annotations = res
        return self._apply_annotations(source_type, annotations)
    origin = get_origin(obj)
    if origin is not None:
        return self._match_generic_type(obj, origin)
    if self._arbitrary_types:
        return self._arbitrary_type_schema(obj)
    return self._unknown_type_schema(obj)