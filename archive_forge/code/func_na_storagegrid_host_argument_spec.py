from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
def na_storagegrid_host_argument_spec():
    return dict(api_url=dict(required=True, type='str'), validate_certs=dict(required=False, type='bool', default=True), auth_token=dict(required=True, type='str', no_log=True))