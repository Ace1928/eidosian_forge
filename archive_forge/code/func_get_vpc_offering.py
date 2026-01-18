from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_vpc_offering(self, key=None):
    vpc_offering = self.module.params.get('vpc_offering')
    args = {'state': 'Enabled'}
    if vpc_offering:
        args['name'] = vpc_offering
        fail_msg = 'VPC offering not found or not enabled: %s' % vpc_offering
    else:
        args['isdefault'] = True
        fail_msg = 'No enabled default VPC offering found'
    vpc_offerings = self.query_api('listVPCOfferings', **args)
    if vpc_offerings:
        for vo in vpc_offerings['vpcoffering']:
            if 'name' in args:
                if args['name'] == vo['name']:
                    return self._get_by_key(key, vo)
            else:
                return self._get_by_key(key, vo)
    self.module.fail_json(msg=fail_msg)