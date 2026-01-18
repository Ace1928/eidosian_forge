import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_bytes_decimal(data, schema):
    """Convert decimal.Decimal to bytes"""
    if not isinstance(data, decimal.Decimal):
        return data
    scale = schema.get('scale', 0)
    precision = schema['precision']
    sign, digits, exp = data.as_tuple()
    if len(digits) > precision:
        raise ValueError('The decimal precision is bigger than allowed by schema')
    delta = exp + scale
    if delta < 0:
        raise ValueError('Scale provided in schema does not match the decimal')
    unscaled_datum = 0
    for digit in digits:
        unscaled_datum = unscaled_datum * 10 + digit
    unscaled_datum = 10 ** delta * unscaled_datum
    bytes_req = (unscaled_datum.bit_length() + 8) // 8
    if sign:
        unscaled_datum = -unscaled_datum
    return unscaled_datum.to_bytes(bytes_req, byteorder='big', signed=True)