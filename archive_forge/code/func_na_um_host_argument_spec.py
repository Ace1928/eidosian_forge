from __future__ import (absolute_import, division, print_function)
import logging
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def na_um_host_argument_spec():
    return dict(hostname=dict(required=True, type='str'), username=dict(required=True, type='str'), password=dict(required=True, type='str', no_log=True), validate_certs=dict(required=False, type='bool', default=True), http_port=dict(required=False, type='int'), feature_flags=dict(required=False, type='dict', default=dict()), max_records=dict(required=False, type='int'))