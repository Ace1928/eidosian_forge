from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def notification_to_dict(notification):
    if not notification:
        return dict()
    return dict(send_to_subscription_administrator=notification.email.send_to_subscription_administrator if notification.email else False, send_to_subscription_co_administrators=notification.email.send_to_subscription_co_administrators if notification.email else False, custom_emails=[to_native(e) for e in notification.email.custom_emails or []], webhooks=[to_native(w.service_url) for w in notification.webhooks or []])