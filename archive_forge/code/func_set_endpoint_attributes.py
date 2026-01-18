import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def set_endpoint_attributes(self, endpoint_arn=None, attributes=None):
    """
        The `SetEndpointAttributes` action sets the attributes for an
        endpoint for a device on one of the supported push
        notification services, such as GCM and APNS. For more
        information, see `Using Amazon SNS Mobile Push
        Notifications`_.

        :type endpoint_arn: string
        :param endpoint_arn: EndpointArn used for SetEndpointAttributes action.

        :type attributes: map
        :param attributes:
        A map of the endpoint attributes. Attributes in this map include the
            following:


        + `CustomUserData` -- arbitrary user data to associate with the
              endpoint. SNS does not use this data. The data must be in UTF-8
              format and less than 2KB.
        + `Enabled` -- flag that enables/disables delivery to the endpoint.
              Message Processor will set this to false when a notification
              service indicates to SNS that the endpoint is invalid. Users can
              set it back to true, typically after updating Token.
        + `Token` -- device token, also referred to as a registration id, for
              an app and mobile device. This is returned from the notification
              service when an app and mobile device are registered with the
              notification service.

        """
    params = {}
    if endpoint_arn is not None:
        params['EndpointArn'] = endpoint_arn
    if attributes is not None:
        self._build_dict_as_list_params(params, attributes, 'Attributes')
    return self._make_request(action='SetEndpointAttributes', params=params)