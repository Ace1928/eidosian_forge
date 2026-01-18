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
def check_lowercase_bucketname(n):
    """
    Bucket names must not contain uppercase characters. We check for
    this by appending a lowercase character and testing with islower().
    Note this also covers cases like numeric bucket names with dashes.

    >>> check_lowercase_bucketname("Aaaa")
    Traceback (most recent call last):
    ...
    BotoClientError: S3Error: Bucket names cannot contain upper-case
    characters when using either the sub-domain or virtual hosting calling
    format.

    >>> check_lowercase_bucketname("1234-5678-9123")
    True
    >>> check_lowercase_bucketname("abcdefg1234")
    True
    """
    if not (n + 'a').islower():
        raise BotoClientError('Bucket names cannot contain upper-case characters when using either the sub-domain or virtual hosting calling format.')
    return True