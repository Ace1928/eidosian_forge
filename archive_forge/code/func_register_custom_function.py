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
def register_custom_function(uri: URIRef, func: _CustomFunction, override: bool=False, raw: bool=False) -> None:
    """
    Register a custom SPARQL function.

    By default, the function will be passed the RDF terms in the argument list.
    If raw is True, the function will be passed an Expression and a Context.

    The function must return an RDF term, or raise a SparqlError.
    """
    if not override and uri in _CUSTOM_FUNCTIONS:
        raise ValueError('A function is already registered as %s' % uri.n3())
    _CUSTOM_FUNCTIONS[uri] = (func, raw)