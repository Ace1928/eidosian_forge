import re
import sys
from pprint import pprint
def is_float_list(value, min=None, max=None):
    """
    Check that the value is a list of floats.
    
    You can optionally specify the minimum and maximum number of members.
    
    Each list member is checked that it is a float.
    
    >>> vtor.check('float_list', ())
    []
    >>> vtor.check('float_list', [])
    []
    >>> vtor.check('float_list', (1, 2.0))
    [1.0, 2.0]
    >>> vtor.check('float_list', [1, 2.0])
    [1.0, 2.0]
    >>> vtor.check('float_list', [1, 'a'])
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    """
    return [is_float(mem) for mem in is_list(value, min, max)]