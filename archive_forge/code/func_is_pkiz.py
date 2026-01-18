import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def is_pkiz(token_text):
    """Determine if a token is PKIZ.

    Checks if the string has the prefix that indicates it is a
    Crypto Message Syntax, Z compressed token.
    """
    return token_text.startswith(PKIZ_PREFIX)