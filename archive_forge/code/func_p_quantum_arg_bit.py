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
def p_quantum_arg_bit(self, p):
    """qarg : ID '[' NATURAL_NUMBER ']'"""
    reg = p[1]
    idx = p[3]
    arg_name = self.make_name(idx, reg)
    if reg not in self.qregs.keys():
        raise QasmException(f'Undefined quantum register "{reg}" at line {p.lineno(1)}')
    size = self.qregs[reg]
    if idx >= size:
        raise QasmException(f'Out of bounds qubit index {idx} on register {reg} of size {size} at line {p.lineno(1)}')
    if arg_name not in self.qubits.keys():
        self.qubits[arg_name] = NamedQubit(arg_name)
    p[0] = [self.qubits[arg_name]]