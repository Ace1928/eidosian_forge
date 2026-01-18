from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def associate_ip_address(self, ip_address):
    self.result['changed'] = True
    args = {'ipaddress': self.module.params.get('ip_address'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'networkid': self.get_network(key='id') if not self.module.params.get('vpc') else None, 'zoneid': self.get_zone(key='id'), 'vpcid': self.get_vpc(key='id')}
    ip_address = None
    if not self.module.check_mode:
        res = self.query_api('associateIpAddress', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            ip_address = self.poll_job(res, 'ipaddress')
    return ip_address