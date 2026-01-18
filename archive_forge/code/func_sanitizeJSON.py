from __future__ import (absolute_import, division, print_function)
import logging
import json
import socket
from uuid import getnode
from ansible.plugins.callback import CallbackBase
from ansible.parsing.ajson import AnsibleJSONEncoder
def sanitizeJSON(self, data):
    try:
        return json.loads(json.dumps(data, sort_keys=True, cls=AnsibleJSONEncoder))
    except Exception:
        return {'warnings': ['JSON Formatting Issue', json.dumps(data, sort_keys=True, cls=AnsibleJSONEncoder)]}