from boto.dynamodb.batch import BatchList
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb import exceptions as dynamodb_exceptions
import time
@classmethod
def create_from_schema(cls, layer2, name, schema):
    """Create a Table object.

        If you know the name and schema of your table, you can
        create a ``Table`` object without having to make any
        API calls (normally an API call is made to retrieve
        the schema of a table).

        Example usage::

            table = Table.create_from_schema(
                boto.connect_dynamodb(),
                'tablename',
                Schema.create(hash_key=('keyname', 'N')))

        :type layer2: :class:`boto.dynamodb.layer2.Layer2`
        :param layer2: A ``Layer2`` api object.

        :type name: str
        :param name: The name of the table.

        :type schema: :class:`boto.dynamodb.schema.Schema`
        :param schema: The schema associated with the table.

        :rtype: :class:`boto.dynamodb.table.Table`
        :return: A Table object representing the table.

        """
    table = cls(layer2, {'Table': {'TableName': name}})
    table._schema = schema
    return table