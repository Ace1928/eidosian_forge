import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def delete_platform_application(self, platform_application_arn=None):
    """
        The `DeletePlatformApplication` action deletes a platform
        application object for one of the supported push notification
        services, such as APNS and GCM. For more information, see
        `Using Amazon SNS Mobile Push Notifications`_.

        :type platform_application_arn: string
        :param platform_application_arn: PlatformApplicationArn of platform
            application object to delete.

        """
    params = {}
    if platform_application_arn is not None:
        params['PlatformApplicationArn'] = platform_application_arn
    return self._make_request(action='DeletePlatformApplication', params=params)