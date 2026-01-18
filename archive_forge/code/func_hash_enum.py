from enum import Enum as PyEnum
from graphene.utils.subclass_with_meta import SubclassWithMeta_Meta
from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType
def hash_enum(self):
    return hash(self.name)