from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_network_acl(self, key=None):
    if self.network_acl is None:
        args = {'name': self.module.params.get('network_acl'), 'vpcid': self.get_vpc(key='id')}
        network_acls = self.query_api('listNetworkACLLists', **args)
        if network_acls:
            self.network_acl = network_acls['networkacllist'][0]
            self.result['network_acl'] = self.network_acl['name']
    if self.network_acl:
        return self._get_by_key(key, self.network_acl)
    else:
        self.fail_json(msg='Network ACL %s not found' % self.module.params.get('network_acl'))