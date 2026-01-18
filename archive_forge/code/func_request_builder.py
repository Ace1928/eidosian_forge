from __future__ import (absolute_import, division, print_function)
import os
import hashlib
import json
import socket
import struct
import traceback
import uuid
from functools import partial
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import cPickle
def request_builder(method_, *args, **kwargs):
    reqid = str(uuid.uuid4())
    req = {'jsonrpc': '2.0', 'method': method_, 'id': reqid}
    req['params'] = (args, kwargs)
    return req