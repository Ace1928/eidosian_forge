from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
def calculate_fingerprint(openssh_publickey):
    digest = hashes.Hash(hashes.SHA256(), backend=backend)
    decoded_pubkey = b64decode(openssh_publickey.split(b' ')[1])
    digest.update(decoded_pubkey)
    return 'SHA256:%s' % b64encode(digest.finalize()).decode(encoding=_TEXT_ENCODING).rstrip('=')