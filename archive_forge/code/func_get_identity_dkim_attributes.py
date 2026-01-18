import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def get_identity_dkim_attributes(self, identities):
    """Get attributes associated with a list of verified identities.

        Given a list of verified identities (email addresses and/or domains),
        returns a structure describing identity notification attributes.

        :type identities: list
        :param identities: A list of verified identities (email addresses
            and/or domains).

        """
    params = {}
    self._build_list_params(params, identities, 'Identities.member')
    return self._make_request('GetIdentityDkimAttributes', params)