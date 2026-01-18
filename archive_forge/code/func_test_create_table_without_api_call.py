from tests.unit import unittest
from mock import Mock
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.table import Table, Schema
def test_create_table_without_api_call(self):
    table = self.layer2.table_from_schema(name='footest', schema=Schema.create(hash_key=('foo', 'N')))
    self.assertEqual(table.name, 'footest')
    self.assertEqual(table.schema, Schema.create(hash_key=('foo', 'N')))
    self.assertEqual(self.api.describe_table.call_count, 0)