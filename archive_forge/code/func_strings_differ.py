import warnings
from webob.compat import (
from webob.headers import _trans_key
def strings_differ(string1, string2, compare_digest=compare_digest):
    """Check whether two strings differ while avoiding timing attacks.

    This function returns True if the given strings differ and False
    if they are equal.  It's careful not to leak information about *where*
    they differ as a result of its running time, which can be very important
    to avoid certain timing-related crypto attacks:

        http://seb.dbzteam.org/crypto/python-oauth-timing-hmac.pdf

    .. versionchanged:: 1.5
       Support :func:`hmac.compare_digest` if it is available (Python 2.7.7+
       and Python 3.3+).

    """
    len_eq = len(string1) == len(string2)
    if len_eq:
        invalid_bits = 0
        left = string1
    else:
        invalid_bits = 1
        left = string2
    right = string2
    if compare_digest is not None:
        invalid_bits += not compare_digest(left, right)
    else:
        for a, b in zip(left, right):
            invalid_bits += a != b
    return invalid_bits != 0