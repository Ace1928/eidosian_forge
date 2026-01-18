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
def setLanguage(terms: Tuple[Any, OptionalType[str]]) -> rdflib.Literal:
    return rdflib.Literal(terms[0], lang=terms[1])