import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_fixed_decimal(data, schema):
    """Converts decimal.Decimal to fixed length bytes array"""
    if not isinstance(data, decimal.Decimal):
        return data
    scale = schema.get('scale', 0)
    size = schema['size']
    precision = schema['precision']
    sign, digits, exp = data.as_tuple()
    if len(digits) > precision:
        raise ValueError('The decimal precision is bigger than allowed by schema')
    if -exp > scale:
        raise ValueError('Scale provided in schema does not match the decimal')
    delta = exp + scale
    if delta > 0:
        digits = digits + (0,) * delta
    unscaled_datum = 0
    for digit in digits:
        unscaled_datum = unscaled_datum * 10 + digit
    bits_req = unscaled_datum.bit_length() + 1
    size_in_bits = size * 8
    offset_bits = size_in_bits - bits_req
    mask = 2 ** size_in_bits - 1
    bit = 1
    for i in range(bits_req):
        mask ^= bit
        bit <<= 1
    if bits_req < 8:
        bytes_req = 1
    else:
        bytes_req = bits_req // 8
        if bits_req % 8 != 0:
            bytes_req += 1
    tmp = BytesIO()
    if sign:
        unscaled_datum = (1 << bits_req) - unscaled_datum
        unscaled_datum = mask | unscaled_datum
        for index in range(size - 1, -1, -1):
            bits_to_write = unscaled_datum >> 8 * index
            tmp.write(bytes([bits_to_write & 255]))
    else:
        for i in range(offset_bits // 8):
            tmp.write(bytes([0]))
        for index in range(bytes_req - 1, -1, -1):
            bits_to_write = unscaled_datum >> 8 * index
            tmp.write(bytes([bits_to_write & 255]))
    return tmp.getvalue()