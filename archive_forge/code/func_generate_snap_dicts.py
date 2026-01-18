from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('snapshots')
def generate_snap_dicts(module, fusion):
    snap_dict = {}
    vsnap_dict = {}
    tenant_api_instance = purefusion.TenantsApi(fusion)
    tenantspace_api_instance = purefusion.TenantSpacesApi(fusion)
    snap_api_instance = purefusion.SnapshotsApi(fusion)
    vsnap_api_instance = purefusion.VolumeSnapshotsApi(fusion)
    tenants = tenant_api_instance.list_tenants()
    for tenant in tenants.items:
        tenant_spaces = tenantspace_api_instance.list_tenant_spaces(tenant_name=tenant.name).items
        for tenant_space in tenant_spaces:
            snaps = snap_api_instance.list_snapshots(tenant_name=tenant.name, tenant_space_name=tenant_space.name)
            for snap in snaps.items:
                snap_name = tenant.name + '/' + tenant_space.name + '/' + snap.name
                secs, mins, hours = _convert_microseconds(snap.time_remaining)
                snap_dict[snap_name] = {'display_name': snap.display_name, 'protection_policy': snap.protection_policy, 'time_remaining': '{0} hours, {1} mins, {2} secs'.format(int(hours), int(mins), int(secs)), 'volume_snapshots_link': snap.volume_snapshots_link}
                vsnaps = vsnap_api_instance.list_volume_snapshots(tenant_name=tenant.name, tenant_space_name=tenant_space.name, snapshot_name=snap.name)
                for vsnap in vsnaps.items:
                    vsnap_name = tenant.name + '/' + tenant_space.name + '/' + snap.name + '/' + vsnap.name
                    secs, mins, hours = _convert_microseconds(vsnap.time_remaining)
                    vsnap_dict[vsnap_name] = {'size': vsnap.size, 'display_name': vsnap.display_name, 'protection_policy': vsnap.protection_policy, 'serial_number': vsnap.serial_number, 'created_at': time.strftime('%a, %d %b %Y %H:%M:%S %Z', time.localtime(vsnap.created_at / 1000)), 'time_remaining': '{0} hours, {1} mins, {2} secs'.format(int(hours), int(mins), int(secs)), 'placement_group': vsnap.placement_group.name}
    return (snap_dict, vsnap_dict)