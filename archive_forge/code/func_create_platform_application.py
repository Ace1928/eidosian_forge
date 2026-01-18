import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def create_platform_application(self, name=None, platform=None, attributes=None):
    """
        The `CreatePlatformApplication` action creates a platform
        application object for one of the supported push notification
        services, such as APNS and GCM, to which devices and mobile
        apps may register. You must specify PlatformPrincipal and
        PlatformCredential attributes when using the
        `CreatePlatformApplication` action. The PlatformPrincipal is
        received from the notification service. For APNS/APNS_SANDBOX,
        PlatformPrincipal is "SSL certificate". For GCM,
        PlatformPrincipal is not applicable. For ADM,
        PlatformPrincipal is "client id". The PlatformCredential is
        also received from the notification service. For
        APNS/APNS_SANDBOX, PlatformCredential is "private key". For
        GCM, PlatformCredential is "API key". For ADM,
        PlatformCredential is "client secret". The
        PlatformApplicationArn that is returned when using
        `CreatePlatformApplication` is then used as an attribute for
        the `CreatePlatformEndpoint` action. For more information, see
        `Using Amazon SNS Mobile Push Notifications`_.

        :type name: string
        :param name: Application names must be made up of only uppercase and
            lowercase ASCII letters, numbers, underscores, hyphens, and
            periods, and must be between 1 and 256 characters long.

        :type platform: string
        :param platform: The following platforms are supported: ADM (Amazon
            Device Messaging), APNS (Apple Push Notification Service),
            APNS_SANDBOX, and GCM (Google Cloud Messaging).

        :type attributes: map
        :param attributes: For a list of attributes, see
            `SetPlatformApplicationAttributes`_

        """
    params = {}
    if name is not None:
        params['Name'] = name
    if platform is not None:
        params['Platform'] = platform
    if attributes is not None:
        self._build_dict_as_list_params(params, attributes, 'Attributes')
    return self._make_request(action='CreatePlatformApplication', params=params)