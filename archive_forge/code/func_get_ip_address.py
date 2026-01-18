from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_ip_address(self, key=None):
    if self.ip_address:
        return self._get_by_key(key, self.ip_address)
    ip_address = self.module.params.get('ip_address')
    if not ip_address:
        self.fail_json(msg="IP address param 'ip_address' is required")
    args = {'ipaddress': ip_address, 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'vpcid': self.get_vpc(key='id')}
    ip_addresses = self.query_api('listPublicIpAddresses', **args)
    if not ip_addresses:
        self.fail_json(msg="IP address '%s' not found" % args['ipaddress'])
    self.ip_address = ip_addresses['publicipaddress'][0]
    return self._get_by_key(key, self.ip_address)