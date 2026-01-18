import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def extract_method_header(headers):
    """
    Extracts the request method from the headers list.
    """
    for k, v in headers:
        if k in (b':method', u':method'):
            if not isinstance(v, bytes):
                return v.encode('utf-8')
            else:
                return v