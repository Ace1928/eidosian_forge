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
def get_api_profile(self, client_type_name, api_profile_name):
    profile_all_clients = AZURE_API_PROFILES.get(api_profile_name)
    if not profile_all_clients:
        raise KeyError('unknown Azure API profile: {0}'.format(api_profile_name))
    profile_raw = profile_all_clients.get(client_type_name, None)
    if not profile_raw:
        self.module.warn('Azure API profile {0} does not define an entry for {1}'.format(api_profile_name, client_type_name))
    if isinstance(profile_raw, dict):
        if not profile_raw.get('default_api_version'):
            raise KeyError("Azure API profile {0} does not define 'default_api_version'".format(api_profile_name))
        return profile_raw
    return dict(default_api_version=profile_raw)