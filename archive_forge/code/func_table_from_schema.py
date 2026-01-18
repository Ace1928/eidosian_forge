from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def table_from_schema(self, name, schema):
    """
        Create a Table object from a schema.

        This method will create a Table object without
        making any API calls.  If you know the name and schema
        of the table, you can use this method instead of
        ``get_table``.

        Example usage::

            table = layer2.table_from_schema(
                'tablename',
                Schema.create(hash_key=('foo', 'N')))

        :type name: str
        :param name: The name of the table.

        :type schema: :class:`boto.dynamodb.schema.Schema`
        :param schema: The schema associated with the table.

        :rtype: :class:`boto.dynamodb.table.Table`
        :return: A Table object representing the table.

        """
    return Table.create_from_schema(self, name, schema)