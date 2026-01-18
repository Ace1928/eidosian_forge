from tests.unit import unittest
from boto.dynamodb.batch import Batch
from boto.dynamodb.table import Table
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.batch import BatchList
def test_batch_consistent_read_defaults_to_false(self):
    b = Batch(self.table, ['k1'])
    self.assertDictEqual(b.to_dict(), {'Keys': [{'HashKeyElement': {'S': 'k1'}}], 'ConsistentRead': False})