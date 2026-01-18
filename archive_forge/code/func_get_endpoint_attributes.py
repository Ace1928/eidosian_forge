import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def get_endpoint_attributes(self, endpoint_arn=None):
    """
        The `GetEndpointAttributes` retrieves the endpoint attributes
        for a device on one of the supported push notification
        services, such as GCM and APNS. For more information, see
        `Using Amazon SNS Mobile Push Notifications`_.

        :type endpoint_arn: string
        :param endpoint_arn: EndpointArn for GetEndpointAttributes input.

        """
    params = {}
    if endpoint_arn is not None:
        params['EndpointArn'] = endpoint_arn
    return self._make_request(action='GetEndpointAttributes', params=params)