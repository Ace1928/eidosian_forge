from mock import patch, Mock
import unittest
import time
from boto.exception import S3ResponseError
from boto.s3.connection import S3Connection
from boto.s3.bucketlogging import BucketLogging
from boto.s3.lifecycle import Lifecycle
from boto.s3.lifecycle import Transition
from boto.s3.lifecycle import Expiration
from boto.s3.lifecycle import Rule
from boto.s3.acl import Grant
from boto.s3.tagging import Tags, TagSet
from boto.s3.website import RedirectLocation
from boto.compat import unquote_str
def test_lifecycle_jp(self):
    name = 'Japanese files'
    prefix = '日本語/'
    days = 30
    lifecycle = Lifecycle()
    lifecycle.add_rule(name, prefix, 'Enabled', days)
    self.bucket.configure_lifecycle(lifecycle)
    readlifecycle = self.bucket.get_lifecycle_config()
    for rule in readlifecycle:
        self.assertEqual(rule.id, name)
        self.assertEqual(rule.expiration.days, days)