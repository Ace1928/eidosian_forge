import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def serialize_num(val):
    """Cast a number to a string and perform
       validation to ensure no loss of precision.
    """
    if isinstance(val, bool):
        return str(int(val))
    return str(val)