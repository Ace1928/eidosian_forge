import functools
import operator
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Union, TYPE_CHECKING
import numpy as np
import sympy
from ply import yacc
from cirq import ops, Circuit, NamedQubit, CX
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import._lexer import QasmLexer
from cirq.contrib.qasm_import.exception import QasmException
def p_new_reg(self, p):
    """new_reg : QREG ID '[' NATURAL_NUMBER ']' ';'
        | CREG ID '[' NATURAL_NUMBER ']' ';'"""
    name, length = (p[2], p[4])
    if name in self.qregs.keys() or name in self.cregs.keys():
        raise QasmException(f'{name} is already defined at line {p.lineno(2)}')
    if length == 0:
        raise QasmException(f"Illegal, zero-length register '{name}' at line {p.lineno(4)}")
    if p[1] == 'qreg':
        self.qregs[name] = length
    else:
        self.cregs[name] = length
    p[0] = (name, length)