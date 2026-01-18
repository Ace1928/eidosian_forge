from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('tenant_spaces')
def generate_ts_dict(module, fusion):
    ts_info = {}
    tenant_api_instance = purefusion.TenantsApi(fusion)
    tenantspace_api_instance = purefusion.TenantSpacesApi(fusion)
    tenants = tenant_api_instance.list_tenants()
    for tenant in tenants.items:
        tenant_spaces = tenantspace_api_instance.list_tenant_spaces(tenant_name=tenant.name).items
        for tenant_space in tenant_spaces:
            ts_name = tenant.name + '/' + tenant_space.name
            ts_info[ts_name] = {'tenant': tenant.name, 'display_name': tenant_space.display_name}
    return ts_info