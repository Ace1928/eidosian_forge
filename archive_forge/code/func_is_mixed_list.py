import re
import sys
from pprint import pprint
def is_mixed_list(value, *args):
    """
    Check that the value is a list.
    Allow specifying the type of each member.
    Work on lists of specific lengths.
    
    You specify each member as a positional argument specifying type
    
    Each type should be one of the following strings :
      'integer', 'float', 'ip_addr', 'string', 'boolean'
    
    So you can specify a list of two strings, followed by
    two integers as :
    
      mixed_list('string', 'string', 'integer', 'integer')
    
    The length of the list must match the number of positional
    arguments you supply.
    
    >>> mix_str = "mixed_list('integer', 'float', 'ip_addr', 'string', 'boolean')"
    >>> check_res = vtor.check(mix_str, (1, 2.0, '1.2.3.4', 'a', True))
    >>> check_res == [1, 2.0, '1.2.3.4', 'a', True]
    1
    >>> check_res = vtor.check(mix_str, ('1', '2.0', '1.2.3.4', 'a', 'True'))
    >>> check_res == [1, 2.0, '1.2.3.4', 'a', True]
    1
    >>> vtor.check(mix_str, ('b', 2.0, '1.2.3.4', 'a', True))
    Traceback (most recent call last):
    VdtTypeError: the value "b" is of the wrong type.
    >>> vtor.check(mix_str, (1, 2.0, '1.2.3.4', 'a'))
    Traceback (most recent call last):
    VdtValueTooShortError: the value "(1, 2.0, '1.2.3.4', 'a')" is too short.
    >>> vtor.check(mix_str, (1, 2.0, '1.2.3.4', 'a', 1, 'b'))
    Traceback (most recent call last):
    VdtValueTooLongError: the value "(1, 2.0, '1.2.3.4', 'a', 1, 'b')" is too long.
    >>> vtor.check(mix_str, 0)
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.

    >>> vtor.check('mixed_list("yoda")', ('a'))
    Traceback (most recent call last):
    VdtParamError: passed an incorrect value "KeyError('yoda',)" for parameter "'mixed_list'"
    """
    try:
        length = len(value)
    except TypeError:
        raise VdtTypeError(value)
    if length < len(args):
        raise VdtValueTooShortError(value)
    elif length > len(args):
        raise VdtValueTooLongError(value)
    try:
        return [fun_dict[arg](val) for arg, val in zip(args, value)]
    except KeyError as e:
        raise VdtParamError('mixed_list', e)