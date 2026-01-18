import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def set_identity_feedback_forwarding_enabled(self, identity, forwarding_enabled=True):
    """
        Enables or disables SES feedback notification via email.
        Feedback forwarding may only be disabled when both complaint and
        bounce topics are set.

        :type identity: string
        :param identity: An email address or domain name.

        :type forwarding_enabled: bool
        :param forwarding_enabled: Specifies whether or not to enable feedback forwarding.
        """
    return self._make_request('SetIdentityFeedbackForwardingEnabled', {'Identity': identity, 'ForwardingEnabled': 'true' if forwarding_enabled else 'false'})