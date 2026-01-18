import os
import sys
from enum import Enum, _simple_enum
def getnode():
    """Get the hardware address as a 48-bit positive integer.

    The first time this runs, it may launch a separate program, which could
    be quite slow.  If all attempts to obtain the hardware address fail, we
    choose a random 48-bit number with its eighth bit set to 1 as recommended
    in RFC 4122.
    """
    global _node
    if _node is not None:
        return _node
    for getter in _GETTERS + [_random_getnode]:
        try:
            _node = getter()
        except:
            continue
        if _node is not None and 0 <= _node < 1 << 48:
            return _node
    assert False, '_random_getnode() returned invalid value: {}'.format(_node)