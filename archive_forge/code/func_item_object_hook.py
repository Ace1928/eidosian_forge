import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def item_object_hook(dct):
    """
    A custom object hook for use when decoding JSON item bodys.
    This hook will transform Amazon DynamoDB JSON responses to something
    that maps directly to native Python types.
    """
    if len(dct.keys()) > 1:
        return dct
    if 'S' in dct:
        return dct['S']
    if 'N' in dct:
        return convert_num(dct['N'])
    if 'SS' in dct:
        return set(dct['SS'])
    if 'NS' in dct:
        return set(map(convert_num, dct['NS']))
    if 'B' in dct:
        return convert_binary(dct['B'])
    if 'BS' in dct:
        return set(map(convert_binary, dct['BS']))
    return dct