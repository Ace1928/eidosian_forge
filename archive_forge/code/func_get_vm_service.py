from __future__ import (absolute_import, division, print_function)
import json
import os
import subprocess
import time
import traceback
import inspect
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def get_vm_service(connection, module):
    if module.params.get('vm_id') is not None or (module.params.get('vm_name') is not None and module.params['state'] != 'absent'):
        vms_service = connection.system_service().vms_service()
        vm_id = module.params['vm_id']
        if vm_id is None:
            vm_id = get_id_by_name(vms_service, module.params['vm_name'])
        if vm_id is None:
            module.fail_json(msg="VM doesn't exist, please create it first.")
        return vms_service.vm_service(vm_id)