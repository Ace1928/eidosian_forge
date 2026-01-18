from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
def compare_encryption_algorithms(ea1, ea2):
    if isinstance(ea1, serialization.NoEncryption) and isinstance(ea2, serialization.NoEncryption):
        return True
    elif isinstance(ea1, serialization.BestAvailableEncryption) and isinstance(ea2, serialization.BestAvailableEncryption):
        return ea1.password == ea2.password
    else:
        return False