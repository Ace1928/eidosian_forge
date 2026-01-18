from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def enum_value(self):
    """Return the value of an enum constant."""
    if not hasattr(self, '_enum_value'):
        assert self.kind == CursorKind.ENUM_CONSTANT_DECL
        underlying_type = self.type
        if underlying_type.kind == TypeKind.ENUM:
            underlying_type = underlying_type.get_declaration().enum_type
        if underlying_type.kind in (TypeKind.CHAR_U, TypeKind.UCHAR, TypeKind.CHAR16, TypeKind.CHAR32, TypeKind.USHORT, TypeKind.UINT, TypeKind.ULONG, TypeKind.ULONGLONG, TypeKind.UINT128):
            self._enum_value = conf.lib.clang_getEnumConstantDeclUnsignedValue(self)
        else:
            self._enum_value = conf.lib.clang_getEnumConstantDeclValue(self)
    return self._enum_value