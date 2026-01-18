from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.exoscale import (ExoDns, exo_dns_argument_spec,
def present_domain(self):
    domain = self.get_domain()
    data = {'domain': {'name': self.name}}
    if not domain:
        self.result['diff']['after'] = data['domain']
        self.result['changed'] = True
        if not self.module.check_mode:
            domain = self.api_query('/domains', 'POST', data)
    return domain