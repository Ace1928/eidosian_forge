from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def set_vrouter_element_state(self, enabled, nsp_name='virtualrouter'):
    vrouter = self.get_vrouter_element(nsp_name)
    if vrouter['enabled'] == enabled:
        return vrouter
    args = {'id': vrouter['id'], 'enabled': enabled}
    if not self.module.check_mode:
        res = self.query_api('configureVirtualRouterElement', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            vrouter = self.poll_job(res, 'virtualrouterelement')
    self.result['changed'] = True
    return vrouter