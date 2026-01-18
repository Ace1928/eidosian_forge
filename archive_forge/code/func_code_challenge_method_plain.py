from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import json
import logging
from oauthlib import common
from .. import errors
from .base import GrantTypeBase
def code_challenge_method_plain(verifier, challenge):
    """
    If the "code_challenge_method" from `Section 4.3`_ was "plain", they are
    compared directly, i.e.:

    code_verifier == code_challenge.

    .. _`Section 4.3`: https://tools.ietf.org/html/rfc7636#section-4.3
    """
    return verifier == challenge