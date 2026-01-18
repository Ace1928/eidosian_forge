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
def expandCollection(terms: ParseResults) -> List[List[Any]]:
    """
    expand ( 1 2 3 ) notation for collections
    """
    if DEBUG:
        print('Collection: ', terms)
    res: List[Any] = []
    other = []
    for x in terms:
        if isinstance(x, list):
            other += x
            x = x[0]
        b = rdflib.BNode()
        if res:
            res += [res[-3], rdflib.RDF.rest, b, b, rdflib.RDF.first, x]
        else:
            res += [b, rdflib.RDF.first, x]
    res += [b, rdflib.RDF.rest, rdflib.RDF.nil]
    res += other
    if DEBUG:
        print('CollectionOut', res)
    return [res]