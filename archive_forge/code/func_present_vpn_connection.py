from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_vpn_connection(self):
    vpn_conn = self.get_vpn_connection()
    args = {'s2scustomergatewayid': self.get_vpn_customer_gateway(key='id'), 's2svpngatewayid': self.get_vpn_gateway(key='id'), 'passive': self.module.params.get('passive')}
    if not vpn_conn:
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('createVpnConnection', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                vpn_conn = self.poll_job(res, 'vpnconnection')
    return vpn_conn