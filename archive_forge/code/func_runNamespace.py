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
def runNamespace() -> str:
    """Returns a URI suitable as a namespace for run-local objects"""
    global runNamespaceValue
    if runNamespaceValue is None:
        runNamespaceValue = join(base(), _unique_id()) + '#'
    return runNamespaceValue