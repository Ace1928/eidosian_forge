from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def safe_string_equals(a, b):
    """ Near-constant time string comparison.

    Used in order to avoid timing attacks on sensitive information such
    as secret keys during request verification (`rootLabs`_).

    .. _`rootLabs`:
    http://rdist.root.org/2010/01/07/timing-independent-array-comparison/

    """
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0