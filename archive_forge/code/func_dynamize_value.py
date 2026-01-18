import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def dynamize_value(val):
    """
    Take a scalar Python value and return a dict consisting
    of the Amazon DynamoDB type specification and the value that
    needs to be sent to Amazon DynamoDB.  If the type of the value
    is not supported, raise a TypeError
    """
    dynamodb_type = get_dynamodb_type(val)
    if dynamodb_type == 'N':
        val = {dynamodb_type: serialize_num(val)}
    elif dynamodb_type == 'S':
        val = {dynamodb_type: val}
    elif dynamodb_type == 'NS':
        val = {dynamodb_type: list(map(serialize_num, val))}
    elif dynamodb_type == 'SS':
        val = {dynamodb_type: [n for n in val]}
    elif dynamodb_type == 'B':
        if isinstance(val, bytes):
            val = Binary(val)
        val = {dynamodb_type: val.encode()}
    elif dynamodb_type == 'BS':
        val = {dynamodb_type: [n.encode() for n in val]}
    return val