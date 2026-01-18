import re
from typing import Optional
import numpy as np
import ply.lex as lex
from cirq.contrib.qasm_import.exception import QasmException
def t_NUMBER(self, t):
    """(
        (
        [0-9]+\\.?|
        [0-9]?\\.[0-9]+
        )
        [eE][+-]?[0-9]+
        )|
        (
        ([0-9]+)?\\.[0-9]+|
        [0-9]+\\.)"""
    t.value = float(t.value)
    return t