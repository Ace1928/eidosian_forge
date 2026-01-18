from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('tenants')
def generate_tenant_dict(module, fusion):
    tenants_api_instance = purefusion.TenantsApi(fusion)
    return {tenant.name: {'display_name': tenant.display_name} for tenant in tenants_api_instance.list_tenants().items}