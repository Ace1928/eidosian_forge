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
def expandBNodeTriples(terms: ParseResults) -> List[Any]:
    """
    expand [ ?p ?o ] syntax for implicit bnodes
    """
    try:
        if DEBUG:
            print('Bnode terms', terms)
            print('1', terms[0])
            print('2', [rdflib.BNode()] + terms.asList()[0])
        return [expandTriples([rdflib.BNode()] + terms.asList()[0])]
    except Exception as e:
        if DEBUG:
            print('>>>>>>>>', e)
        raise