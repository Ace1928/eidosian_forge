import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def eval_with_catch(expr, vars):
    try:
        return eval(expr, vars)
    except Exception as e:
        _add_except(e, 'in expression %r' % expr)
        raise