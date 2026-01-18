import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def set_identity_dkim_enabled(self, identity, dkim_enabled):
    """Enables or disables DKIM signing of email sent from an identity.

        * If Easy DKIM signing is enabled for a domain name identity (e.g.,
        * ``example.com``),
          then Amazon SES will DKIM-sign all email sent by addresses under that
          domain name (e.g., ``user@example.com``)
        * If Easy DKIM signing is enabled for an email address, then Amazon SES
          will DKIM-sign all email sent by that email address.

        For email addresses (e.g., ``user@example.com``), you can only enable
        Easy DKIM signing  if the corresponding domain (e.g., ``example.com``)
        has been set up for Easy DKIM using the AWS Console or the
        ``VerifyDomainDkim`` action.

        :type identity: string
        :param identity: An email address or domain name.

        :type dkim_enabled: bool
        :param dkim_enabled: Specifies whether or not to enable DKIM signing.

        """
    return self._make_request('SetIdentityDkimEnabled', {'Identity': identity, 'DkimEnabled': 'true' if dkim_enabled else 'false'})