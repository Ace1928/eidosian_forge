import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def cms_sign_text(data_to_sign, signing_cert_file_name, signing_key_file_name, message_digest=DEFAULT_TOKEN_DIGEST_ALGORITHM):
    return cms_sign_data(data_to_sign, signing_cert_file_name, signing_key_file_name, message_digest=message_digest)