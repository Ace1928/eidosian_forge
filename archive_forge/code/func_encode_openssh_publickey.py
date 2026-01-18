from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
@staticmethod
def encode_openssh_publickey(asym_keypair, comment):
    """Returns an OpenSSH encoded public key for a given keypair

           :asym_keypair: Asymmetric_Keypair from the public key is extracted
           :comment: Comment to apply to the end of the returned OpenSSH encoded public key
        """
    encoded_publickey = asym_keypair.public_key.public_bytes(encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH)
    validate_comment(comment)
    encoded_publickey += (' %s' % comment).encode(encoding=_TEXT_ENCODING) if comment else b''
    return encoded_publickey