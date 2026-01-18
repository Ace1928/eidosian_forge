from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
def extract_comment(path):
    if not os.path.exists(path):
        raise InvalidPublicKeyFileError('No file was found at %s' % path)
    try:
        with open(path, 'rb') as f:
            fields = f.read().split(b' ', 2)
            if len(fields) == 3:
                comment = fields[2].decode(_TEXT_ENCODING)
            else:
                comment = ''
    except (IOError, OSError) as e:
        raise InvalidPublicKeyFileError(e)
    return comment