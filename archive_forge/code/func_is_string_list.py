import re
import sys
from pprint import pprint
def is_string_list(value, min=None, max=None):
    """
    Check that the value is a list of strings.
    
    You can optionally specify the minimum and maximum number of members.
    
    Each list member is checked that it is a string.
    
    >>> vtor.check('string_list', ())
    []
    >>> vtor.check('string_list', [])
    []
    >>> vtor.check('string_list', ('a', 'b'))
    ['a', 'b']
    >>> vtor.check('string_list', ['a', 1])
    Traceback (most recent call last):
    VdtTypeError: the value "1" is of the wrong type.
    >>> vtor.check('string_list', 'hello')
    Traceback (most recent call last):
    VdtTypeError: the value "hello" is of the wrong type.
    """
    if isinstance(value, string_type):
        raise VdtTypeError(value)
    return [is_string(mem) for mem in is_list(value, min, max)]