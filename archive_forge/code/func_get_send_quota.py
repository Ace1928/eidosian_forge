import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def get_send_quota(self):
    """Fetches the user's current activity limits.

        :rtype: dict
        :returns: A GetSendQuotaResponse structure. Note that keys must be
                  unicode strings.
        """
    return self._make_request('GetSendQuota')