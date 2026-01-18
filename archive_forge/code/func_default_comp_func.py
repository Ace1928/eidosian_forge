import hashlib
import importlib
import json
import re
import urllib.parse
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.parsing.convert_bool import boolean
def default_comp_func(device):
    for k in uniquity_keys:
        if not params.get(k):
            continue
        if isinstance(device, dict):
            v = device['value'].get(k)
        elif isinstance(device, list):
            v = device
        else:
            exceptions = importlib.import_module('ansible_collections.cloud.common.plugins.module_utils.turbo.exceptions')
            raise exceptions.EmbeddedModuleFailure(msg='Unexpect type')
        if isinstance(k, int) or isinstance(v, str):
            k = str(k)
            v = str(v)
        if v == params.get(k):
            return device