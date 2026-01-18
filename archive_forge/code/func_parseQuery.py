from __future__ import annotations
import re
import sys
from typing import Any, BinaryIO, List
from typing import Optional as OptionalType
from typing import TextIO, Tuple, Union
from pyparsing import CaselessKeyword as Keyword  # watch out :)
from pyparsing import (
import rdflib
from rdflib.compat import decodeUnicodeEscape
from . import operators as op
from .parserutils import Comp, CompValue, Param, ParamList
def parseQuery(q: Union[str, bytes, TextIO, BinaryIO]) -> ParseResults:
    if hasattr(q, 'read'):
        q = q.read()
    if isinstance(q, bytes):
        q = q.decode('utf-8')
    q = expandUnicodeEscapes(q)
    return Query.parseString(q, parseAll=True)