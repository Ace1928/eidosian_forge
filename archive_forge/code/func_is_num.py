import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def is_num(n, boolean_as_int=True):
    if boolean_as_int:
        types = (int, long_type, float, Decimal, bool)
    else:
        types = (int, long_type, float, Decimal)
    return isinstance(n, types) or n in types