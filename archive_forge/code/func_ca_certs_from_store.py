import os
import ssl
import sys
from ... import config
from ... import version_string as breezy_version
def ca_certs_from_store(path):
    if not os.path.exists(path):
        raise ValueError('ca certs path %s does not exist' % path)
    return path