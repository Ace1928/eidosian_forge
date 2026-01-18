from __future__ import annotations
import datetime as py_datetime  # naming conflict with function within this module
import hashlib
import math
import operator as pyop  # python operators
import random
import re
import uuid
import warnings
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from functools import reduce
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union, overload
from urllib.parse import quote
import isodate
from pyparsing import ParseResults
from rdflib.namespace import RDF, XSD
from rdflib.plugins.sparql.datatypes import (
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import (
def numeric(expr: Literal) -> Any:
    """
    return a number from a literal
    http://www.w3.org/TR/xpath20/#promotion

    or TypeError
    """
    if not isinstance(expr, Literal):
        raise SPARQLTypeError('%r is not a literal!' % expr)
    if expr.datatype not in (XSD.float, XSD.double, XSD.decimal, XSD.integer, XSD.nonPositiveInteger, XSD.negativeInteger, XSD.nonNegativeInteger, XSD.positiveInteger, XSD.unsignedLong, XSD.unsignedInt, XSD.unsignedShort, XSD.unsignedByte, XSD.long, XSD.int, XSD.short, XSD.byte):
        raise SPARQLTypeError('%r does not have a numeric datatype!' % expr)
    return expr.toPython()