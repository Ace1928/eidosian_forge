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
@staticmethod
def create_enum(graphene_type):
    values = {}
    for name, value in graphene_type._meta.enum.__members__.items():
        description = getattr(value, 'description', None)
        if isinstance(description, PyEnum):
            description = None
        if not description and callable(graphene_type._meta.description):
            description = graphene_type._meta.description(value)
        deprecation_reason = getattr(value, 'deprecation_reason', None)
        if isinstance(deprecation_reason, PyEnum):
            deprecation_reason = None
        if not deprecation_reason and callable(graphene_type._meta.deprecation_reason):
            deprecation_reason = graphene_type._meta.deprecation_reason(value)
        values[name] = GraphQLEnumValue(value=value, description=description, deprecation_reason=deprecation_reason)
    type_description = graphene_type._meta.description(None) if callable(graphene_type._meta.description) else graphene_type._meta.description
    return GrapheneEnumType(graphene_type=graphene_type, values=values, name=graphene_type._meta.name, description=type_description)