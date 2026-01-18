import xml.sax
import base64
import time
from boto.compat import six, urllib
from boto.auth import detect_potential_s3sigv4
import boto.utils
from boto.connection import AWSAuthConnection
from boto import handler
from boto.s3.bucket import Bucket
from boto.s3.key import Key
from boto.resultset import ResultSet
from boto.exception import BotoClientError, S3ResponseError
from boto.utils import get_utf8able_str
def get_canonical_user_id(self, headers=None):
    """
        Convenience method that returns the "CanonicalUserID" of the
        user who's credentials are associated with the connection.
        The only way to get this value is to do a GET request on the
        service which returns all buckets associated with the account.
        As part of that response, the canonical userid is returned.
        This method simply does all of that and then returns just the
        user id.

        :rtype: string
        :return: A string containing the canonical user id.
        """
    rs = self.get_all_buckets(headers=headers)
    return rs.owner.id