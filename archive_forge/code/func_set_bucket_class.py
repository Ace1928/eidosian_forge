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
def set_bucket_class(self, bucket_class):
    """
        Set the Bucket class associated with this bucket.  By default, this
        would be the boto.s3.key.Bucket class but if you want to subclass that
        for some reason this allows you to associate your new class.

        :type bucket_class: class
        :param bucket_class: A subclass of Bucket that can be more specific
        """
    self.bucket_class = bucket_class