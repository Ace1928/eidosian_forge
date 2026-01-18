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
def p_params_multiple(self, p):
    """params : expr ',' params"""
    p[3].insert(0, p[1])
    p[0] = p[3]