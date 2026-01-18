import pytest
from numpy.f2py.symbolic import (
from . import util
def collect_symbols(s):
    if s.op is Op.APPLY:
        oper = s.data[0]
        function_symbols.add(oper)
        if oper in symbols:
            symbols.remove(oper)
    elif s.op is Op.SYMBOL and s not in function_symbols:
        symbols.add(s)