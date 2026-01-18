import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def handle_unprocessed(self, resp):
    if len(resp.get('UnprocessedItems', [])):
        table_name = self.table.table_name
        unprocessed = resp['UnprocessedItems'].get(table_name, [])
        msg = '%s items were unprocessed. Storing for later.'
        boto.log.info(msg % len(unprocessed))
        self._unprocessed.extend(unprocessed)