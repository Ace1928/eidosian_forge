from tests.unit import unittest
from boto.dynamodb.batch import Batch
from boto.dynamodb.table import Table
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.batch import BatchList
def test_batch_list_consistent_read(self):
    b = BatchList(self.layer2)
    b.add_batch(self.table, ['k1'], ['foo'], consistent_read=True)
    b.add_batch(self.table2, [('k2', 54)], ['bar'], consistent_read=False)
    self.assertDictEqual(b.to_dict(), {'testtable': {'AttributesToGet': ['foo'], 'Keys': [{'HashKeyElement': {'S': 'k1'}}], 'ConsistentRead': True}, 'testtable2': {'AttributesToGet': ['bar'], 'Keys': [{'HashKeyElement': {'S': 'k2'}, 'RangeKeyElement': {'N': '54'}}], 'ConsistentRead': False}})