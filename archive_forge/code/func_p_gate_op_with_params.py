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
def p_gate_op_with_params(self, p):
    """gate_op :  ID '(' params ')' qargs"""
    self._resolve_gate_operation(args=p[5], gate=p[1], p=p, params=p[3])