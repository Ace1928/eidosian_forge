from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_vpn_connection(self):
    args = {'vpcid': self.get_vpc(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
    vpn_conns = self.query_api('listVpnConnections', **args)
    if vpn_conns:
        for vpn_conn in vpn_conns['vpnconnection']:
            if self.get_vpn_customer_gateway(key='id') == vpn_conn['s2scustomergatewayid']:
                return vpn_conn