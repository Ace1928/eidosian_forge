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
def build_post_policy(self, expiration_time, conditions):
    """
        Taken from the AWS book Python examples and modified for use with boto
        """
    assert isinstance(expiration_time, time.struct_time), 'Policy document must include a valid expiration Time object'
    return '{"expiration": "%s",\n"conditions": [%s]}' % (time.strftime(boto.utils.ISO8601, expiration_time), ','.join(conditions))