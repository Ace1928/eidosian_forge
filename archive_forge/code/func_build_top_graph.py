import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def build_top_graph(self, tokens):
    """Build a DotGraph instance from parsed data"""
    strict = tokens[0] == 'strict'
    graphtype = tokens[1]
    directed = graphtype == 'digraph'
    graphname = tokens[2]
    graph = DotGraph(graphname, strict, directed)
    self.graph = self.build_graph(graph, tokens[3])