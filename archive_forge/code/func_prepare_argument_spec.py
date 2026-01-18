from __future__ import absolute_import, division, print_function
import json
import socket
from ansible.module_utils.basic import env_fallback
from ansible_collections.vmware.vmware_rest.plugins.module_utils.vmware_rest import (
def prepare_argument_spec():
    argument_spec = {'vcenter_hostname': dict(type='str', required=True, fallback=(env_fallback, ['VMWARE_HOST'])), 'vcenter_username': dict(type='str', required=True, fallback=(env_fallback, ['VMWARE_USER'])), 'vcenter_password': dict(type='str', required=True, no_log=True, fallback=(env_fallback, ['VMWARE_PASSWORD'])), 'vcenter_validate_certs': dict(type='bool', required=False, default=True, fallback=(env_fallback, ['VMWARE_VALIDATE_CERTS'])), 'vcenter_rest_log_file': dict(type='str', required=False, fallback=(env_fallback, ['VMWARE_REST_LOG_FILE'])), 'session_timeout': dict(type='float', required=False, fallback=(env_fallback, ['VMWARE_SESSION_TIMEOUT']))}
    return argument_spec