import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def pkiz_verify(signed_text, signing_cert_file_name, ca_file_name):
    uncompressed = pkiz_uncompress(signed_text)
    return cms_verify(uncompressed, signing_cert_file_name, ca_file_name, inform=PKIZ_CMS_FORM)