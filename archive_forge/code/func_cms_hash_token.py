import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def cms_hash_token(token_id, mode='md5'):
    """Hash PKI tokens.

    return: for asn1 or pkiz tokens, returns the hash of the passed in token
            otherwise, returns what it was passed in.
    """
    if token_id is None:
        return None
    if is_asn1_token(token_id) or is_pkiz(token_id):
        hasher = hashlib.new(mode)
        if isinstance(token_id, str):
            token_id = token_id.encode('utf-8')
        hasher.update(token_id)
        return hasher.hexdigest()
    else:
        return token_id