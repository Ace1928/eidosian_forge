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
def checkDot(self, argstr: str, i: int) -> int:
    j = self.skipSpace(argstr, i)
    if j < 0:
        return j
    ch = argstr[j]
    if ch == '.':
        return j + 1
    if ch == '}':
        return j
    if ch == ']':
        return j
    self.BadSyntax(argstr, j, "expected '.' or '}' or ']' at end of statement")