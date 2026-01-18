from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def generate_nonce():
    """Generate pseudorandom nonce that is unlikely to repeat.

    Per `section 3.3`_ of the OAuth 1 RFC 5849 spec.
    Per `section 3.2.1`_ of the MAC Access Authentication spec.

    A random 64-bit number is appended to the epoch timestamp for both
    randomness and to decrease the likelihood of collisions.

    .. _`section 3.2.1`:
    https://tools.ietf.org/html/draft-ietf-oauth-v2-http-mac-01#section-3.2.1
    .. _`section 3.3`: https://tools.ietf.org/html/rfc5849#section-3.3
    """
    return unicode_type(unicode_type(randbits(64)) + generate_timestamp())