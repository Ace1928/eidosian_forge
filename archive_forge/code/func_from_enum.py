from enum import Enum as PyEnum
from graphene.utils.subclass_with_meta import SubclassWithMeta_Meta
from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType
def from_enum(cls, enum, name=None, description=None, deprecation_reason=None):
    name = name or enum.__name__
    description = description or enum.__doc__ or 'An enumeration.'
    meta_dict = {'enum': enum, 'description': description, 'deprecation_reason': deprecation_reason}
    meta_class = type('Meta', (object,), meta_dict)
    return type(name, (Enum,), {'Meta': meta_class})