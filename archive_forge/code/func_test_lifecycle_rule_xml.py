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
def test_lifecycle_rule_xml(self):
    rule = Rule(status='Enabled', expiration=30)
    s = rule.to_xml()
    self.assertEqual(s.find('<ID>'), -1)
    self.assertNotEqual(s.find('<Prefix></Prefix>'), -1)