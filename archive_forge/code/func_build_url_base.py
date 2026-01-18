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
def build_url_base(self, connection, protocol, server, bucket, key=''):
    url_base = '//'
    url_base += self.build_host(server, bucket)
    url_base += connection.get_path(self.build_path_base(bucket, key))
    return url_base