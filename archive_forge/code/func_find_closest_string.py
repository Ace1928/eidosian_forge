import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
def find_closest_string(string, strings):

    def _key(s):
        return (levenshtein(s, string), s)
    return sorted(strings, key=_key)[0]