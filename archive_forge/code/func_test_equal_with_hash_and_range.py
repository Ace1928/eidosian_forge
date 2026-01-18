from tests.unit import unittest
from mock import Mock
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.table import Table, Schema
def test_equal_with_hash_and_range(self):
    s1 = Schema.create(hash_key=('foo', 'N'), range_key=('bar', 'S'))
    s2 = Schema.create(hash_key=('foo', 'N'), range_key=('bar', 'S'))
    self.assertEqual(s1, s2)