from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def dynamize_scan_filter(self, scan_filter):
    """
        Convert a layer2 scan_filter parameter into the
        structure required by Layer1.
        """
    d = None
    if scan_filter:
        d = {}
        for attr_name in scan_filter:
            condition = scan_filter[attr_name]
            d[attr_name] = condition.to_dict()
    return d