from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def dynamize_item(self, item):
    d = {}
    for attr_name in item:
        d[attr_name] = self.dynamizer.encode(item[attr_name])
    return d