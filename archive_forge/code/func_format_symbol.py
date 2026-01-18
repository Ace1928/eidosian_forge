import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def format_symbol(symbol):
    """Returns well formatted Hermann-Mauguin symbol as extected by
    the database, by correcting the case and adding missing or
    removing dublicated spaces."""
    fixed = []
    s = symbol.strip()
    s = s[0].upper() + s[1:].lower()
    for c in s:
        if c.isalpha():
            if len(fixed) and fixed[-1] == '/':
                fixed.append(c)
            else:
                fixed.append(' ' + c + ' ')
        elif c.isspace():
            fixed.append(' ')
        elif c.isdigit():
            fixed.append(c)
        elif c == '-':
            fixed.append(' ' + c)
        elif c == '/':
            fixed.append(c)
    s = ''.join(fixed).strip()
    return ' '.join(s.split())