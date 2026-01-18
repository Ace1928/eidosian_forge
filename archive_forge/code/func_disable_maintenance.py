from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def disable_maintenance(self, host):
    if host['resourcestate'] in ['PrepareForMaintenance', 'Maintenance']:
        self.result['changed'] = True
        args = {'id': host['id']}
        if not self.module.check_mode:
            res = self.query_api('cancelHostMaintenance', **args)
            host = self.poll_job(res, 'host')
    return host