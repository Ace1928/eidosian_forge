import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
@staticmethod
def build_regex_replacement(regex_str):
    match = re.match('\\^(\\w)\\(\\[2\\-9\\]\\|\\[1\\-9\\]\\[0\\-9\\]\\+\\)\\?\\( domain\\)\\?\\$', regex_str)
    if match:
        anchor_char = match.group(1)
        return ('^' + anchor_char + '1$', anchor_char)
    else:
        return None