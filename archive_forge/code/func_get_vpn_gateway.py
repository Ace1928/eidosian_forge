from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_vpn_gateway(self, key=None):
    args = {'vpcid': self.get_vpc(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
    vpn_gateways = self.query_api('listVpnGateways', **args)
    if vpn_gateways:
        return self._get_by_key(key, vpn_gateways['vpngateway'][0])
    elif self.module.params.get('force'):
        if self.module.check_mode:
            return {}
        res = self.query_api('createVpnGateway', **args)
        vpn_gateway = self.poll_job(res, 'vpngateway')
        return self._get_by_key(key, vpn_gateway)
    self.fail_json(msg='VPN gateway not found and not forced to create one')