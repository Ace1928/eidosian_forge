from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
def hexify(ustr: str) -> bytes:
    """Use URL encoding to return an ASCII string
    corresponding to the given UTF8 string

    >>> hexify("http://example/a b")
    b'http://example/a%20b'

    """
    s = ''
    for ch in ustr:
        if ord(ch) > 126 or ord(ch) < 33:
            ch = '%%%02X' % ord(ch)
        else:
            ch = '%c' % ord(ch)
        s = s + ch
    return s.encode('latin-1')