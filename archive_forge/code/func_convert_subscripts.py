import itertools
from collections import OrderedDict
import numpy as np
def convert_subscripts(old_sub, symbol_map):
    """Convert user custom subscripts list to subscript string according to `symbol_map`.

    Examples
    --------
    >>>  oe.parser.convert_subscripts(['abc', 'def'], {'abc':'a', 'def':'b'})
    'ab'
    >>> oe.parser.convert_subscripts([Ellipsis, object], {object:'a'})
    '...a'
    """
    new_sub = ''
    for s in old_sub:
        if s is Ellipsis:
            new_sub += '...'
        else:
            new_sub += symbol_map[s]
    return new_sub