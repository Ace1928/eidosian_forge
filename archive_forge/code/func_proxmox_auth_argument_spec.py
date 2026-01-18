from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def proxmox_auth_argument_spec():
    return dict(api_host=dict(type='str', required=True, fallback=(env_fallback, ['PROXMOX_HOST'])), api_user=dict(type='str', required=True, fallback=(env_fallback, ['PROXMOX_USER'])), api_password=dict(type='str', no_log=True, fallback=(env_fallback, ['PROXMOX_PASSWORD'])), api_token_id=dict(type='str', no_log=False), api_token_secret=dict(type='str', no_log=True), validate_certs=dict(type='bool', default=False))