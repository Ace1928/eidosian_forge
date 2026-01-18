import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def get_vault_notifications(self, vault_name):
    """
        This operation retrieves the `notification-configuration`
        subresource of the specified vault.

        For information about setting a notification configuration on
        a vault, see SetVaultNotifications. If a notification
        configuration for a vault is not set, the operation returns a
        `404 Not Found` error. For more information about vault
        notifications, see `Configuring Vault Notifications in Amazon
        Glacier`_.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and underlying REST API, go to
        `Configuring Vault Notifications in Amazon Glacier`_ and `Get
        Vault Notification Configuration `_ in the Amazon Glacier
        Developer Guide .

        :type vault_name: string
        :param vault_name: The name of the vault.
        """
    uri = 'vaults/%s/notification-configuration' % vault_name
    return self.make_request('GET', uri)