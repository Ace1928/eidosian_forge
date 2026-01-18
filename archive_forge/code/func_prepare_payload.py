import hashlib
import importlib
import json
import re
import urllib.parse
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.parsing.convert_bool import boolean
def prepare_payload(params, payload_format):
    payload = {}
    for i in payload_format['body'].keys():
        if params[i] is None:
            continue
        path = payload_format['body'][i]
        set_subkey(payload, path, params[i])
    return payload