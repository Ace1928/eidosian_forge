import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_date(obj):
    """Attempt to convert to utc-iso time string using date methods."""
    try:
        time_string = obj.isoformat()
    except AttributeError:
        raise NotEncodable
    else:
        return iso_to_plotly_time_string(time_string)