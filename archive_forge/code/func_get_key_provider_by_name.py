from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
@staticmethod
def get_key_provider_by_name(key_provider_clusters, name):
    key_provider_cluster = None
    if not name or not key_provider_clusters:
        return key_provider_cluster
    for kp_cluster in key_provider_clusters:
        if kp_cluster.clusterId.id == name:
            key_provider_cluster = kp_cluster
    return key_provider_cluster