from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_argument_spec():
    """Return standard base dictionary used for the argument_spec
    argument in AnsibleModule

    """
    return dict(api_key=dict(type='str', aliases=['password'], no_log=True), auth_endpoint=dict(type='str'), credentials=dict(type='path', aliases=['creds_file']), env=dict(type='str'), identity_type=dict(type='str', default='rackspace'), region=dict(type='str'), tenant_id=dict(type='str'), tenant_name=dict(type='str'), username=dict(type='str'), validate_certs=dict(type='bool', aliases=['verify_ssl']))