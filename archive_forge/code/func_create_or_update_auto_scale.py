from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
from datetime import timedelta
def create_or_update_auto_scale(self, param):
    try:
        return self.monitor_autoscale_settings_client.autoscale_settings.create_or_update(self.resource_group, self.name, param)
    except Exception as exc:
        self.fail('Error creating auto scale settings {0} - {1}'.format(self.name, str(exc)))