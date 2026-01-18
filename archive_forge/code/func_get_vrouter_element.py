from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_vrouter_element(self, nsp_name='virtualrouter'):
    nsp = self.get_nsp(nsp_name)
    nspid = nsp['id']
    if self.vrouters is None:
        self.vrouters = dict()
        res = self.query_api('listVirtualRouterElements')
        for vrouter in res['virtualrouterelement']:
            self.vrouters[vrouter['nspid']] = vrouter
    if nspid not in self.vrouters:
        self.module.fail_json(msg="Failed: No VirtualRouterElement found for nsp '%s'" % nsp_name)
    return self.vrouters[nspid]