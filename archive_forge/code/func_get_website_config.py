import boto
import os
import sys
import textwrap
from boto.s3.deletemarker import DeleteMarker
from boto.exception import BotoClientError
from boto.exception import InvalidUriError
def get_website_config(self, validate=False, headers=None):
    self._check_bucket_uri('get_website_config')
    bucket = self.get_bucket(validate, headers)
    return bucket.get_website_configuration(headers)