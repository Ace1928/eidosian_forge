from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def dynamize_last_evaluated_key(self, last_evaluated_key):
    """
        Convert a last_evaluated_key parameter into the data structure
        required for Layer1.
        """
    d = None
    if last_evaluated_key:
        hash_key = last_evaluated_key['HashKeyElement']
        d = {'HashKeyElement': self.dynamizer.encode(hash_key)}
        if 'RangeKeyElement' in last_evaluated_key:
            range_key = last_evaluated_key['RangeKeyElement']
            d['RangeKeyElement'] = self.dynamizer.encode(range_key)
    return d