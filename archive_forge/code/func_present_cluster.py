from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_cluster(self):
    cluster = self.get_cluster()
    if cluster:
        cluster = self._update_cluster()
    else:
        cluster = self._create_cluster()
    return cluster