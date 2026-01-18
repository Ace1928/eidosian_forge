import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def set_identity_notification_topic(self, identity, notification_type, sns_topic=None):
    """Sets an SNS topic to publish bounce or complaint notifications for
        emails sent with the given identity as the Source. Publishing to topics
        may only be disabled when feedback forwarding is enabled.

        :type identity: string
        :param identity: An email address or domain name.

        :type notification_type: string
        :param notification_type: The type of feedback notifications that will
                                  be published to the specified topic.
                                  Valid Values: Bounce | Complaint | Delivery

        :type sns_topic: string or None
        :param sns_topic: The Amazon Resource Name (ARN) of the Amazon Simple
                          Notification Service (Amazon SNS) topic.
        """
    params = {'Identity': identity, 'NotificationType': notification_type}
    if sns_topic:
        params['SnsTopic'] = sns_topic
    return self._make_request('SetIdentityNotificationTopic', params)