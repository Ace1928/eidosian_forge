import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def list_platform_applications(self, next_token=None):
    """
        The `ListPlatformApplications` action lists the platform
        application objects for the supported push notification
        services, such as APNS and GCM. The results for
        `ListPlatformApplications` are paginated and return a limited
        list of applications, up to 100. If additional records are
        available after the first page results, then a NextToken
        string will be returned. To receive the next page, you call
        `ListPlatformApplications` using the NextToken string received
        from the previous call. When there are no more records to
        return, NextToken will be null. For more information, see
        `Using Amazon SNS Mobile Push Notifications`_.

        :type next_token: string
        :param next_token: NextToken string is used when calling
            ListPlatformApplications action to retrieve additional records that
            are available after the first page results.

        """
    params = {}
    if next_token is not None:
        params['NextToken'] = next_token
    return self._make_request(action='ListPlatformApplications', params=params)