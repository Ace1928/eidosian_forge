from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def check_galaxy_version(schema):
    params = _load_params()
    if not params:
        return
    params_keys = list(params.keys())
    if 'method' in params_keys and 'method' not in schema:
        error_message = 'Legacy playbook detected, please revise the playbook or install latest legacy'
        error_message += ' fortimanager galaxy collection: #ansible-galaxy collection install -f fortinet.fortimanager:1.0.5'
        raise Exception(error_message)