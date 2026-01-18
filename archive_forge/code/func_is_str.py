import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def is_str(n):
    return isinstance(n, str) or (isinstance(n, type) and issubclass(n, str))