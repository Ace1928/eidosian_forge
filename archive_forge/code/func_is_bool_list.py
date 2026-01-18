import re
import sys
from pprint import pprint
def is_bool_list(value, min=None, max=None):
    """
    Check that the value is a list of booleans.
    
    You can optionally specify the minimum and maximum number of members.
    
    Each list member is checked that it is a boolean.
    
    >>> vtor.check('bool_list', ())
    []
    >>> vtor.check('bool_list', [])
    []
    >>> check_res = vtor.check('bool_list', (True, False))
    >>> check_res == [True, False]
    1
    >>> check_res = vtor.check('bool_list', [True, False])
    >>> check_res == [True, False]
    1
    >>> vtor.check('bool_list', [True, 'a'])
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    """
    return [is_boolean(mem) for mem in is_list(value, min, max)]