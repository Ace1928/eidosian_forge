from __future__ import absolute_import, division, print_function
import json
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.common.text.converters import to_native
def build_headers(api_token):
    """Takes api token, returns headers with it included."""
    headers = {'X-Circonus-App-Name': 'ansible', 'Host': 'api.circonus.com', 'X-Circonus-Auth-Token': api_token, 'Accept': 'application/json'}
    return headers