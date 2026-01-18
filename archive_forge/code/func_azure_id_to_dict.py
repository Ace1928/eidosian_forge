from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def azure_id_to_dict(id):
    pieces = re.sub('^\\/', '', id).split('/')
    result = {}
    index = 0
    while index < len(pieces) - 1:
        result[pieces[index]] = pieces[index + 1]
        index += 1
    return result