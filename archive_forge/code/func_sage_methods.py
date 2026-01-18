import sys
import doctest
import re
import types
from .numeric_output_checker import NumericOutputChecker
def sage_methods(obj):
    ans = []
    for attr in dir(obj):
        try:
            methods = getattr(obj, attr)
            if methods._sage_method is True:
                ans.append(methods)
        except AttributeError:
            pass
    return ans