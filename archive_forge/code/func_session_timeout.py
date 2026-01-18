import hashlib
import importlib
import json
import re
import urllib.parse
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.parsing.convert_bool import boolean
def session_timeout(params):
    exceptions = importlib.import_module('ansible_collections.cloud.common.plugins.module_utils.turbo.exceptions')
    try:
        aiohttp = importlib.import_module('aiohttp')
    except ImportError:
        raise exceptions.EmbeddedModuleFailure(msg=missing_required_lib('aiohttp'))
    if not aiohttp:
        raise exceptions.EmbeddedModuleFailure(msg='Failed to import aiohttp')
    out = {}
    if params.get('session_timeout'):
        out['timeout'] = aiohttp.ClientTimeout(total=params.get('session_timeout'))
    return out