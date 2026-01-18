import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_med(med):
    """Returns True if value of *med* is valid as per RFC.

    According to RFC MED is a four octet non-negative integer and
    value '((2 ** 32) - 1) =  0xffffffff' denotes an "infinity" metric.
    """
    return isinstance(med, numbers.Integral) and 0 <= med <= 4294967295