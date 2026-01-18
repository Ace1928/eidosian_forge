from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
@staticmethod
def encode_certificates(certificate_file):
    """
        Read certificate file and encode it
        """
    try:
        with open(certificate_file, mode='rb') as fh:
            cert = fh.read()
    except (OSError, IOError) as exc:
        return (None, str(exc))
    if not cert:
        return (None, 'Error: file is empty')
    return (base64.b64encode(cert).decode('utf-8'), None)