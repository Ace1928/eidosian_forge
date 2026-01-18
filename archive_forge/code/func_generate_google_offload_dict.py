from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_google_offload_dict(array):
    offload_info = {}
    offloads = list(array.get_offloads(protocol='google-cloud').items)
    for offload in range(0, len(offloads)):
        name = offloads[offload].name
        offload_info[name] = {'snapshots': getattr(offloads[offload].space, 'snapshots', None), 'shared': getattr(offloads[offload].space, 'shared', None), 'data_reduction': getattr(offloads[offload].space, 'data_reduction', None), 'thin_provisioning': getattr(offloads[offload].space, 'thin_provisioning', None), 'total_physical': getattr(offloads[offload].space, 'total_physical', None), 'total_provisioned': getattr(offloads[offload].space, 'total_provisioned', None), 'total_reduction': getattr(offloads[offload].space, 'total_reduction', None), 'unique': getattr(offloads[offload].space, 'unique', None), 'virtual': getattr(offloads[offload].space, 'virtual', None), 'replication': getattr(offloads[offload].space, 'replication', None), 'used_provisioned': getattr(offloads[offload].space, 'used_provisioned', None)}
        if LooseVersion(SUBS_API_VERSION) <= LooseVersion(array.get_rest_version()):
            offload_info[name]['total_used'] = offloads[offload].space.total_used
    return offload_info