import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
@removals.remove(message='Use is_asn1_token() instead.', version='1.7.0', removal_version='2.0.0')
def is_ans1_token(token):
    """Deprecated.

    This function is deprecated as of the 1.7.0 release in favor of
    :func:`is_asn1_token` and may be removed in the 2.0.0 release.
    """
    return is_asn1_token(token)