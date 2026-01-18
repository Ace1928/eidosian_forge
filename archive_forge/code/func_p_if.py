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
def p_if(self, p):
    """if : IF '(' carg EQ NATURAL_NUMBER ')' gate_op"""
    conditions = []
    for i, key in enumerate(p[3]):
        v = p[5] >> i & 1
        conditions.append(sympy.Eq(sympy.Symbol(key), v))
    p[0] = [ops.ClassicallyControlledOperation(conditions=conditions, sub_operation=tuple(p[7])[0])]