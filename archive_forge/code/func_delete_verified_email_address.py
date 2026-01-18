import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def delete_verified_email_address(self, email_address):
    """Deletes the specified email address from the list of verified
        addresses.

        :type email_adddress: string
        :param email_address: The email address to be removed from the list of
                              verified addreses.

        :rtype: dict
        :returns: A DeleteVerifiedEmailAddressResponse structure. Note that
                  keys must be unicode strings.
        """
    return self._make_request('DeleteVerifiedEmailAddress', {'EmailAddress': email_address})