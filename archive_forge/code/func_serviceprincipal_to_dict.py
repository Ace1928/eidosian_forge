from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def serviceprincipal_to_dict(self, object):
    return dict(app_id=object.app_id, object_id=object.object_id, app_display_name=object.display_name, app_role_assignment_required=object.app_role_assignment_required)