import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def delete_vault_notifications(self, vault_name):
    """
        This operation deletes the notification configuration set for
        a vault. The operation is eventually consistent;that is, it
        might take some time for Amazon Glacier to completely disable
        the notifications and you might still receive some
        notifications for a short time after you send the delete
        request.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and underlying REST API, go to
        `Configuring Vault Notifications in Amazon Glacier`_ and
        `Delete Vault Notification Configuration `_ in the Amazon
        Glacier Developer Guide.

        :type vault_name: string
        :param vault_name: The name of the vault.
        """
    uri = 'vaults/%s/notification-configuration' % vault_name
    return self.make_request('DELETE', uri, ok_responses=(204,))