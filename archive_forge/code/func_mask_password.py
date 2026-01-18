import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def mask_password(message, secret='***'):
    """Replace password with *secret* in message.

    :param message: The string which includes security information.
    :param secret: value with which to replace passwords.
    :returns: The unicode value of message with the password fields masked.

    For example:

    >>> mask_password("'adminPass' : 'aaaaa'")
    "'adminPass' : '***'"
    >>> mask_password("'admin_pass' : 'aaaaa'")
    "'admin_pass' : '***'"
    >>> mask_password('"password" : "aaaaa"')
    '"password" : "***"'
    >>> mask_password("'original_password' : 'aaaaa'")
    "'original_password' : '***'"

    .. versionadded:: 0.2

    .. versionchanged:: 1.1
       Replace also ``'auth_token'``, ``'new_pass'`` and ``'auth_password'``
       keys.

    .. versionchanged:: 1.1.1
       Replace also ``'secret_uuid'`` key.

    .. versionchanged:: 1.5
       Replace also ``'sys_pswd'`` key.

    .. versionchanged:: 2.6
       Replace also ``'token'`` key.

    .. versionchanged:: 2.7
       Replace also ``'secret'`` key.

    .. versionchanged:: 3.4
       Replace also ``'configdrive'`` key.

    .. versionchanged:: 3.8
       Replace also ``'CHAPPASSWORD'`` key.
    """
    try:
        message = str(message)
    except UnicodeDecodeError:
        pass
    substitute1 = '\\g<1>' + secret
    substitute2 = '\\g<1>' + secret + '\\g<2>'
    substitute_wildcard = '\\g<1>'
    for key in _SANITIZE_KEYS:
        if key in message.lower():
            for pattern in _SANITIZE_PATTERNS_2[key]:
                message = re.sub(pattern, substitute2, message)
            for pattern in _SANITIZE_PATTERNS_1[key]:
                message = re.sub(pattern, substitute1, message)
            for pattern in _SANITIZE_PATTERNS_WILDCARD[key]:
                message = re.sub(pattern, substitute_wildcard, message)
    return message