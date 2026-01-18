import re
import dns.exception
def is_metaclass(rdclass):
    """True if the specified class is a metaclass.

    The currently defined metaclasses are ANY and NONE.

    *rdclass* is an ``int``.
    """
    if rdclass in _metaclasses:
        return True
    return False