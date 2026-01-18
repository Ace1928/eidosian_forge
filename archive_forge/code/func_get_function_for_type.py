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
def get_function_for_type(self, graphene_type, func_name, name, default_value):
    """Gets a resolve or subscribe function for a given ObjectType"""
    if not issubclass(graphene_type, ObjectType):
        return
    resolver = getattr(graphene_type, func_name, None)
    if not resolver:
        interface_resolver = None
        for interface in graphene_type._meta.interfaces:
            if name not in interface._meta.fields:
                continue
            interface_resolver = getattr(interface, func_name, None)
            if interface_resolver:
                break
        resolver = interface_resolver
    if resolver:
        return get_unbound_function(resolver)