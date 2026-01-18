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
def bareWord(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
    """abc -> :abc"""
    j = self.skipSpace(argstr, i)
    if j < 0:
        return -1
    if argstr[j] in numberChars or argstr[j] in _notKeywordsChars:
        return -1
    i = j
    len_argstr = len(argstr)
    while i < len_argstr and argstr[i] not in _notKeywordsChars:
        i += 1
    res.append(argstr[j:i])
    return i