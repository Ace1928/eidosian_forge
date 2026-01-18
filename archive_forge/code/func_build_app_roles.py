from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def build_app_roles(self, app_roles):
    if not app_roles:
        return None
    result = []
    if isinstance(app_roles, dict):
        self.log('Getting "appRoles" from a full manifest')
        app_roles = app_roles.get('appRoles', [])
    for x in app_roles:
        role = AppRole(id=x.get('id', None) or self.gen_guid(), allowed_member_types=x.get('allowed_member_types', None), description=x.get('description', None), display_name=x.get('display_name', None), is_enabled=x.get('is_enabled', None), value=x.get('value', None))
        result.append(role)
    return result