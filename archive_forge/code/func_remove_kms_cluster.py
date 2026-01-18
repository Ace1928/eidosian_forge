from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def remove_kms_cluster(self, kp_cluster):
    for kms in kp_cluster.servers:
        self.remove_kms_server(kp_cluster.clusterId, kms.name)