import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def collect_step_tables(self):
    self._lsts = {}
    for step_name in self.ALL_STEPS:
        self._lsts[step_name] = []
    self._seen_struct_unions = set()
    self._generate('ctx')
    self._add_missing_struct_unions()
    for step_name in self.ALL_STEPS:
        lst = self._lsts[step_name]
        if step_name != 'field':
            lst.sort(key=lambda entry: entry.name)
        self._lsts[step_name] = tuple(lst)
    lst = self._lsts['struct_union']
    for tp, i in self._struct_unions.items():
        assert i < len(lst)
        assert lst[i].name == tp.name
    assert len(lst) == len(self._struct_unions)
    lst = self._lsts['enum']
    for tp, i in self._enums.items():
        assert i < len(lst)
        assert lst[i].name == tp.name
    assert len(lst) == len(self._enums)