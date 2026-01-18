from __future__ import (absolute_import, division, print_function)
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
from ansible.module_utils.basic import AnsibleModule
import traceback
def get_all_permits(self):
    return dict(((permit.name, permit.id) for permit in self._connection.system_service().cluster_levels_service().level_service('4.3').get().permits))