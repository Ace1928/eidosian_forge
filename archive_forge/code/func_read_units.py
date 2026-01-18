from boto.dynamodb.batch import BatchList
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb import exceptions as dynamodb_exceptions
import time
@property
def read_units(self):
    try:
        return self._dict['ProvisionedThroughput']['ReadCapacityUnits']
    except KeyError:
        return None