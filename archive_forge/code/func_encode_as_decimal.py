import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_decimal(obj):
    """Attempt to encode decimal by converting it to float"""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    else:
        raise NotEncodable