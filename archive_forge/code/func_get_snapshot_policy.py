from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_snapshot_policy(self):
    args = {'volumeid': self.get_volume(key='id')}
    policies = self.query_api('listSnapshotPolicies', **args)
    if policies:
        for policy in policies['snapshotpolicy']:
            if policy['intervaltype'] == self.get_interval_type():
                return policy
        return None