from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_snapshot_policy(self):
    required_params = ['schedule']
    self.module.fail_on_missing_params(required_params=required_params)
    policy = self.get_snapshot_policy()
    args = {'id': policy.get('id') if policy else None, 'intervaltype': self.module.params.get('interval_type'), 'schedule': self.module.params.get('schedule'), 'maxsnaps': self.module.params.get('max_snaps'), 'timezone': self.module.params.get('time_zone'), 'volumeid': self.get_volume(key='id')}
    if not policy or (policy and self.has_changed(policy, args, only_keys=['schedule', 'maxsnaps', 'timezone'])):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('createSnapshotPolicy', **args)
            policy = res['snapshotpolicy']
    return policy