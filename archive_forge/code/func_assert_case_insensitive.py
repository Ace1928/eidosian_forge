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
def assert_case_insensitive(f):

    def wrapper(*args, **kwargs):
        if len(args) == 3 and check_lowercase_bucketname(args[2]):
            pass
        return f(*args, **kwargs)
    return wrapper