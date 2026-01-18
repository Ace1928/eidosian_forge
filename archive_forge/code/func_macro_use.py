import functools
import re
import nltk.tree
def macro_use(n, m=None, l=None):
    if m is None or macro_name not in m:
        raise TgrepException(f'macro {macro_name} not defined')
    return m[macro_name](n, m, l)