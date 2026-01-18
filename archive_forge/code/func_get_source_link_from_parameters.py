from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def get_source_link_from_parameters(params):
    tenant = params['tenant']
    tenant_space = params['tenant_space']
    volume = params['source_volume']
    snapshot = params['source_snapshot']
    volume_snapshot = params['source_volume_snapshot']
    if tenant is None or tenant_space is None:
        return None
    if volume is not None:
        return f'/tenants/{tenant}/tenant-spaces/{tenant_space}/volumes/{volume}'
    if snapshot is not None and volume_snapshot is not None:
        return f'/tenants/{tenant}/tenant-spaces/{tenant_space}/snapshots/{snapshot}/volume-snapshots/{volume_snapshot}'
    return None