import re
from typing import Optional
import numpy as np
import ply.lex as lex
from cirq.contrib.qasm_import.exception import QasmException
def t_FORMAT_SPEC(self, t):
    """OPENQASM(\\s+)([^\\s\\t\\;]*);"""
    match = re.match('OPENQASM(\\s+)([^\\s\\t;]*);', t.value)
    t.value = match.groups()[1]
    return t