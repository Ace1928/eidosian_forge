import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def get_key_fields(self):
    """
        Returns the fields necessary to make a key for a table.

        If the ``Table`` does not already have a populated ``schema``,
        this will request it via a ``Table.describe`` call.

        Returns a list of fieldnames (strings).

        Example::

            # A simple hash key.
            >>> users.get_key_fields()
            ['username']

            # A complex hash+range key.
            >>> users.get_key_fields()
            ['username', 'last_name']

        """
    if not self.schema:
        self.describe()
    return [field.name for field in self.schema]