from enum import Enum as PyEnum
import inspect
from functools import partial
from graphql import (
from ..utils.str_converters import to_camel_case
from ..utils.get_unbound_function import get_unbound_function
from .definitions import (
from .dynamic import Dynamic
from .enum import Enum
from .field import Field
from .inputobjecttype import InputObjectType
from .interface import Interface
from .objecttype import ObjectType
from .resolver import get_default_resolver
from .scalars import ID, Boolean, Float, Int, Scalar, String
from .structures import List, NonNull
from .union import Union
from .utils import get_field_as
def normalize_execute_kwargs(kwargs):
    """Replace alias names in keyword arguments for graphql()"""
    if 'root' in kwargs and 'root_value' not in kwargs:
        kwargs['root_value'] = kwargs.pop('root')
    if 'context' in kwargs and 'context_value' not in kwargs:
        kwargs['context_value'] = kwargs.pop('context')
    if 'variables' in kwargs and 'variable_values' not in kwargs:
        kwargs['variable_values'] = kwargs.pop('variables')
    if 'operation' in kwargs and 'operation_name' not in kwargs:
        kwargs['operation_name'] = kwargs.pop('operation')
    return kwargs