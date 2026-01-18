import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def collect_type_table(self):
    self._typesdict = {}
    self._generate('collecttype')
    all_decls = sorted(self._typesdict, key=str)
    self.cffi_types = []
    for tp in all_decls:
        if tp.is_raw_function:
            assert self._typesdict[tp] is None
            self._typesdict[tp] = len(self.cffi_types)
            self.cffi_types.append(tp)
            for tp1 in tp.args:
                assert isinstance(tp1, (model.VoidType, model.BasePrimitiveType, model.PointerType, model.StructOrUnionOrEnum, model.FunctionPtrType))
                if self._typesdict[tp1] is None:
                    self._typesdict[tp1] = len(self.cffi_types)
                self.cffi_types.append(tp1)
            self.cffi_types.append('END')
    for tp in all_decls:
        if not tp.is_raw_function and self._typesdict[tp] is None:
            self._typesdict[tp] = len(self.cffi_types)
            self.cffi_types.append(tp)
            if tp.is_array_type and tp.length is not None:
                self.cffi_types.append('LEN')
    assert None not in self._typesdict.values()
    self._struct_unions = {}
    self._enums = {}
    for tp in all_decls:
        if isinstance(tp, model.StructOrUnion):
            self._struct_unions[tp] = None
        elif isinstance(tp, model.EnumType):
            self._enums[tp] = None
    for i, tp in enumerate(sorted(self._struct_unions, key=lambda tp: tp.name)):
        self._struct_unions[tp] = i
    for i, tp in enumerate(sorted(self._enums, key=lambda tp: tp.name)):
        self._enums[tp] = i
    for tp in all_decls:
        method = getattr(self, '_emit_bytecode_' + tp.__class__.__name__)
        method(tp, self._typesdict[tp])
    for op in self.cffi_types:
        assert isinstance(op, CffiOp)
    self.cffi_types = tuple(self.cffi_types)