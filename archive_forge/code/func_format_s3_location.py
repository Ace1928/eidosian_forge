import hashlib
import io
from unittest import mock
import uuid
import boto3
import botocore
from botocore import exceptions as boto_exceptions
from botocore import stub
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import s3
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def format_s3_location(user, key, authurl, bucket, obj):
    """Helper method that returns a S3 store URI given the component pieces."""
    scheme = 's3'
    if authurl.startswith('https://'):
        scheme = 's3+https'
        authurl = authurl[len('https://'):]
    elif authurl.startswith('http://'):
        authurl = authurl[len('http://'):]
    authurl = authurl.strip('/')
    return '%s://%s:%s@%s/%s/%s' % (scheme, user, key, authurl, bucket, obj)