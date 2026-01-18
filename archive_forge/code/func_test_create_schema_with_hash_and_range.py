from tests.unit import unittest
from mock import Mock
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.table import Table, Schema
def test_create_schema_with_hash_and_range(self):
    schema = self.layer2.create_schema('foo', int, 'bar', str)
    self.assertEqual(schema.hash_key_name, 'foo')
    self.assertEqual(schema.hash_key_type, 'N')
    self.assertEqual(schema.range_key_name, 'bar')
    self.assertEqual(schema.range_key_type, 'S')