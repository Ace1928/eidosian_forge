import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def create_global_secondary_index(self, global_index):
    """
        Creates a global index in DynamoDB after the table has been created.

        Requires a ``global_indexes`` parameter, which should be a
        ``GlobalBaseIndexField`` subclass representing the desired index.

        To update ``global_indexes`` information on the ``Table``, you'll need
        to call ``Table.describe``.

        Returns ``True`` on success.

        Example::

            # To create a global index
            >>> users.create_global_secondary_index(
            ...     global_index=GlobalAllIndex(
            ...         'TheIndexNameHere', parts=[
            ...             HashKey('requiredHashkey', data_type=STRING),
            ...             RangeKey('optionalRangeKey', data_type=STRING)
            ...         ],
            ...         throughput={
            ...             'read': 2,
            ...             'write': 1,
            ...         })
            ... )
            True

        """
    if global_index:
        gsi_data = []
        gsi_data_attr_def = []
        gsi_data.append({'Create': global_index.schema()})
        for attr_def in global_index.parts:
            gsi_data_attr_def.append(attr_def.definition())
        self.connection.update_table(self.table_name, global_secondary_index_updates=gsi_data, attribute_definitions=gsi_data_attr_def)
        return True
    else:
        msg = 'You need to provide the global_index to create_global_secondary_index method'
        boto.log.error(msg)
        return False