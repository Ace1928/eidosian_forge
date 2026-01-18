from boto.dynamodb.batch import BatchList
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb import exceptions as dynamodb_exceptions
import time
def has_item(self, hash_key, range_key=None, consistent_read=False):
    """
        Checks the table to see if the Item with the specified ``hash_key``
        exists. This may save a tiny bit of time/bandwidth over a
        straight :py:meth:`get_item` if you have no intention to touch
        the data that is returned, since this method specifically tells
        Amazon not to return anything but the Item's key.

        :type hash_key: int|long|float|str|unicode|Binary
        :param hash_key: The HashKey of the requested item.  The
            type of the value must match the type defined in the
            schema for the table.

        :type range_key: int|long|float|str|unicode|Binary
        :param range_key: The optional RangeKey of the requested item.
            The type of the value must match the type defined in the
            schema for the table.

        :type consistent_read: bool
        :param consistent_read: If True, a consistent read
            request is issued.  Otherwise, an eventually consistent
            request is issued.

        :rtype: bool
        :returns: ``True`` if the Item exists, ``False`` if not.
        """
    try:
        self.get_item(hash_key, range_key=range_key, attributes_to_get=[hash_key], consistent_read=consistent_read)
    except dynamodb_exceptions.DynamoDBKeyNotFoundError:
        return False
    return True