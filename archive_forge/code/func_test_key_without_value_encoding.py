import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_key_without_value_encoding(self):
    input_dict = {'KeyWithNoValue': '', 'AnotherKeyWithNoValue': None}
    res = self.service_connection._build_tag_list(input_dict)
    expected = {'Tags.member.1.Key': 'AnotherKeyWithNoValue', 'Tags.member.2.Key': 'KeyWithNoValue'}
    self.assertEqual(expected, res)