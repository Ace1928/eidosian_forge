from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def gen_scheduled_event_profile(self):
    if self.terminate_event_timeout_minutes is None:
        return None
    scheduledEventProfile = self.compute_models.ScheduledEventsProfile()
    terminationProfile = self.compute_models.TerminateNotificationProfile()
    terminationProfile.not_before_timeout = 'PT' + str(self.terminate_event_timeout_minutes) + 'M'
    terminationProfile.enable = True
    scheduledEventProfile.terminate_notification_profile = terminationProfile
    return scheduledEventProfile