import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_datetime(obj):
    """Convert datetime objects to iso-format strings"""
    try:
        return obj.isoformat()
    except AttributeError:
        raise NotEncodable