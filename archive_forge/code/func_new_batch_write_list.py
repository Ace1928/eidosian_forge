from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def new_batch_write_list(self):
    """
        Return a new, empty :class:`boto.dynamodb.batch.BatchWriteList`
        object.
        """
    return BatchWriteList(self)