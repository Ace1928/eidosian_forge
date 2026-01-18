import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_numpy(obj):
    """Attempt to convert numpy.ma.core.masked"""
    numpy = get_module('numpy', should_load=False)
    if not numpy:
        raise NotEncodable
    if obj is numpy.ma.core.masked:
        return float('nan')
    elif isinstance(obj, numpy.ndarray) and obj.dtype.kind == 'M':
        try:
            return numpy.datetime_as_string(obj).tolist()
        except TypeError:
            pass
    raise NotEncodable