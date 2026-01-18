from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import integer_types, string_types
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import traceback
def auth_argument_spec(spec=None):
    arg_spec = dict(ca_path=dict(type='str'), api_token=dict(type='str', no_log=True), api_oauth_token=dict(type='str', no_log=True), api_job_token=dict(type='str', no_log=True))
    if spec:
        arg_spec.update(spec)
    return arg_spec