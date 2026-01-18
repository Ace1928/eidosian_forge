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
def format_resource_id(val, subscription_id, namespace, types, resource_group):
    return resource_id(name=val, resource_group=resource_group, namespace=namespace, type=types, subscription=subscription_id) if not is_valid_resource_id(val) else val