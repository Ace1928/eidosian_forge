from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def manageiq_argument_spec():
    options = dict(url=dict(default=os.environ.get('MIQ_URL', None)), username=dict(default=os.environ.get('MIQ_USERNAME', None)), password=dict(default=os.environ.get('MIQ_PASSWORD', None), no_log=True), token=dict(default=os.environ.get('MIQ_TOKEN', None), no_log=True), validate_certs=dict(default=True, type='bool', aliases=['verify_ssl']), ca_cert=dict(required=False, default=None, aliases=['ca_bundle_path']))
    return dict(manageiq_connection=dict(type='dict', apply_defaults=True, options=options))