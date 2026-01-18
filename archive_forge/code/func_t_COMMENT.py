import re
from typing import Optional
import numpy as np
import ply.lex as lex
from cirq.contrib.qasm_import.exception import QasmException
def t_COMMENT(self, t):
    """//.*"""