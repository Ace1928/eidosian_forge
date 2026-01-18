import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def set_vault_notifications(self, vault_name, notification_config):
    """
        This operation configures notifications that will be sent when
        specific events happen to a vault. By default, you don't get
        any notifications.

        To configure vault notifications, send a PUT request to the
        `notification-configuration` subresource of the vault. The
        request should include a JSON document that provides an Amazon
        SNS topic and specific events for which you want Amazon
        Glacier to send notifications to the topic.

        Amazon SNS topics must grant permission to the vault to be
        allowed to publish notifications to the topic. You can
        configure a vault to publish a notification for the following
        vault events:


        + **ArchiveRetrievalCompleted** This event occurs when a job
          that was initiated for an archive retrieval is completed
          (InitiateJob). The status of the completed job can be
          "Succeeded" or "Failed". The notification sent to the SNS
          topic is the same output as returned from DescribeJob.
        + **InventoryRetrievalCompleted** This event occurs when a job
          that was initiated for an inventory retrieval is completed
          (InitiateJob). The status of the completed job can be
          "Succeeded" or "Failed". The notification sent to the SNS
          topic is the same output as returned from DescribeJob.


        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and underlying REST API, go to
        `Configuring Vault Notifications in Amazon Glacier`_ and `Set
        Vault Notification Configuration `_ in the Amazon Glacier
        Developer Guide .

        :type vault_name: string
        :param vault_name: The name of the vault.

        :type vault_notification_config: dict
        :param vault_notification_config: Provides options for specifying
            notification configuration.

            The format of the dictionary is:

                {'SNSTopic': 'mytopic',
                 'Events': [event1,...]}
        """
    uri = 'vaults/%s/notification-configuration' % vault_name
    json_config = json.dumps(notification_config)
    return self.make_request('PUT', uri, data=json_config, ok_responses=(204,))