from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def check_placement_policy():
    if self.param('placement_policy'):
        hosts = sorted(map(lambda host: self._connection.follow_link(host).name, entity.placement_policy.hosts if entity.placement_policy.hosts else []))
        if self.param('placement_policy_hosts'):
            return equal(self.param('placement_policy'), str(entity.placement_policy.affinity) if entity.placement_policy else None) and equal(sorted(self.param('placement_policy_hosts')), hosts)
        return equal(self.param('placement_policy'), str(entity.placement_policy.affinity) if entity.placement_policy else None) and equal([self.param('host')], hosts)
    return True