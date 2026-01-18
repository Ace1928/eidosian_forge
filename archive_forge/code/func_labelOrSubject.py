from __future__ import annotations
from typing import Any, MutableSequence
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.parser import InputSource, Parser
from .notation3 import RDFSink, SinkParser
def labelOrSubject(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
    j = self.skipSpace(argstr, i)
    if j < 0:
        return j
    i = j
    j = self.uri_ref2(argstr, i, res)
    if j >= 0:
        return j
    if argstr[i] == '[':
        j = self.skipSpace(argstr, i + 1)
        if j < 0:
            self.BadSyntax(argstr, i, 'Expected ] got EOF')
        if argstr[j] == ']':
            res.append(self.blankNode())
            return j + 1
    return -1