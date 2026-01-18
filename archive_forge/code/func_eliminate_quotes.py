import re
import warnings
from enum import Enum
from math import gcd
def eliminate_quotes(s):
    """Replace quoted substrings of input string.

    Return a new string and a mapping of replacements.
    """
    d = {}

    def repl(m):
        kind, value = m.groups()[:2]
        if kind:
            kind = kind[:-1]
        p = {"'": 'SINGLE', '"': 'DOUBLE'}[value[0]]
        k = f'{kind}@__f2py_QUOTES_{p}_{COUNTER.__next__()}@'
        d[k] = value
        return k
    new_s = re.sub('({kind}_|)({single_quoted}|{double_quoted})'.format(kind='\\w[\\w\\d_]*', single_quoted="('([^'\\\\]|(\\\\.))*')", double_quoted='("([^"\\\\]|(\\\\.))*")'), repl, s)
    assert '"' not in new_s
    assert "'" not in new_s
    return (new_s, d)