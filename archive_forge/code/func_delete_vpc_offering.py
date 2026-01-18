from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def delete_vpc_offering(self):
    vpc_offering = self.get_vpc_offering()
    if vpc_offering:
        self.result['changed'] = True
        args = {'id': vpc_offering['id']}
        if not self.module.check_mode:
            res = self.query_api('deleteVPCOffering', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                vpc_offering = self.poll_job(res, 'vpcoffering')
    return vpc_offering