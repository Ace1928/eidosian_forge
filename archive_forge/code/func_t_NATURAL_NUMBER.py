import re
from typing import Optional
import numpy as np
import ply.lex as lex
from cirq.contrib.qasm_import.exception import QasmException
def t_NATURAL_NUMBER(self, t):
    """\\d+"""
    t.value = int(t.value)
    return t