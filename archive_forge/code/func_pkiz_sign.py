import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def pkiz_sign(text, signing_cert_file_name, signing_key_file_name, compression_level=6, message_digest=DEFAULT_TOKEN_DIGEST_ALGORITHM):
    signed = cms_sign_data(text, signing_cert_file_name, signing_key_file_name, PKIZ_CMS_FORM, message_digest=message_digest)
    compressed = zlib.compress(signed, compression_level)
    encoded = PKIZ_PREFIX + base64.urlsafe_b64encode(compressed).decode('utf-8')
    return encoded