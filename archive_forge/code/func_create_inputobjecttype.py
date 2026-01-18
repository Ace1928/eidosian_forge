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
def create_inputobjecttype(self, graphene_type):
    return GrapheneInputObjectType(graphene_type=graphene_type, name=graphene_type._meta.name, description=graphene_type._meta.description, out_type=graphene_type._meta.container, fields=partial(self.create_fields_for_type, graphene_type, is_input_type=True))