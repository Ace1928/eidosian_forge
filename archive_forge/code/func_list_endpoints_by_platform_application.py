import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def list_endpoints_by_platform_application(self, platform_application_arn=None, next_token=None):
    """
        The `ListEndpointsByPlatformApplication` action lists the
        endpoints and endpoint attributes for devices in a supported
        push notification service, such as GCM and APNS. The results
        for `ListEndpointsByPlatformApplication` are paginated and
        return a limited list of endpoints, up to 100. If additional
        records are available after the first page results, then a
        NextToken string will be returned. To receive the next page,
        you call `ListEndpointsByPlatformApplication` again using the
        NextToken string received from the previous call. When there
        are no more records to return, NextToken will be null. For
        more information, see `Using Amazon SNS Mobile Push
        Notifications`_.

        :type platform_application_arn: string
        :param platform_application_arn: PlatformApplicationArn for
            ListEndpointsByPlatformApplicationInput action.

        :type next_token: string
        :param next_token: NextToken string is used when calling
            ListEndpointsByPlatformApplication action to retrieve additional
            records that are available after the first page results.

        """
    params = {}
    if platform_application_arn is not None:
        params['PlatformApplicationArn'] = platform_application_arn
    if next_token is not None:
        params['NextToken'] = next_token
    return self._make_request(action='ListEndpointsByPlatformApplication', params=params)