from os import environ, path
from sys import platform as _sys_platform
from re import match, split, search, MULTILINE, IGNORECASE
from kivy.compat import string_types
def strtotuple(s):
    """Convert a tuple string into a tuple
    with some security checks. Designed to be used
    with the eval() function::

        a = (12, 54, 68)
        b = str(a)         # return '(12, 54, 68)'
        c = strtotuple(b)  # return (12, 54, 68)

    """
    if not match('^[,.0-9 ()\\[\\]]*$', s):
        raise Exception('Invalid characters in string for tuple conversion')
    if s.count('(') != s.count(')'):
        raise Exception('Invalid count of ( and )')
    if s.count('[') != s.count(']'):
        raise Exception('Invalid count of [ and ]')
    r = eval(s)
    if type(r) not in (list, tuple):
        raise Exception('Conversion failed')
    return r