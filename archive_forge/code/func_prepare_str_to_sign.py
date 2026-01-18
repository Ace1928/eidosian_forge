from __future__ import (absolute_import, division, print_function)
from base64 import b64encode
from email.utils import formatdate
import re
import json
import hashlib
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
def prepare_str_to_sign(req_tgt, hdrs):
    """
    Concatenates Intersight headers in preparation to be signed

    :param req_tgt : http method plus endpoint
    :param hdrs: dict with header keys
    :return: concatenated header authorization string
    """
    ss = ''
    ss = ss + '(request-target): ' + req_tgt + '\n'
    length = len(hdrs.items())
    i = 0
    for key, value in hdrs.items():
        ss = ss + key.lower() + ': ' + value
        if i < length - 1:
            ss = ss + '\n'
        i += 1
    return ss