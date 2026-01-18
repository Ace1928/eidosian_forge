from enum import Enum as PyEnum
from graphene.utils.subclass_with_meta import SubclassWithMeta_Meta
from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType
def eq_enum(self, other):
    if isinstance(other, self.__class__):
        return self is other
    return self.value is other