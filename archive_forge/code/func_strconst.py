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
def strconst(self, argstr: str, i: int, delim: str) -> Tuple[int, str]:
    """parse an N3 string constant delimited by delim.
        return index, val
        """
    delim1 = delim[0]
    delim2, delim3, delim4, delim5 = (delim1 * 2, delim1 * 3, delim1 * 4, delim1 * 5)
    j = i
    ustr = ''
    startline = self.lines
    len_argstr = len(argstr)
    while j < len_argstr:
        if argstr[j] == delim1:
            if delim == delim1:
                i = j + 1
                return (i, ustr)
            if delim == delim3:
                if argstr[j:j + 5] == delim5:
                    i = j + 5
                    ustr += delim2
                    return (i, ustr)
                if argstr[j:j + 4] == delim4:
                    i = j + 4
                    ustr += delim1
                    return (i, ustr)
                if argstr[j:j + 3] == delim3:
                    i = j + 3
                    return (i, ustr)
                j += 1
                ustr += delim1
                continue
        m = interesting.search(argstr, j)
        assert m, 'Quote expected in string at ^ in %s^%s' % (argstr[j - 20:j], argstr[j:j + 20])
        i = m.start()
        try:
            ustr += argstr[j:i]
        except UnicodeError:
            err = ''
            for c in argstr[j:i]:
                err = err + ' %02x' % ord(c)
            streason = sys.exc_info()[1].__str__()
            raise BadSyntax(self._thisDoc, startline, argstr, j, 'Unicode error appending characters' + ' %s to string, because\n\t%s' % (err, streason))
        ch = argstr[i]
        if ch == delim1:
            j = i
            continue
        elif ch in {'"', "'"} and ch != delim1:
            ustr += ch
            j = i + 1
            continue
        elif ch in {'\r', '\n'}:
            if delim == delim1:
                raise BadSyntax(self._thisDoc, startline, argstr, i, 'newline found in string literal')
            self.lines += 1
            ustr += ch
            j = i + 1
            self.startOfLine = j
        elif ch == '\\':
            j = i + 1
            ch = argstr[j]
            if not ch:
                raise BadSyntax(self._thisDoc, startline, argstr, i, 'unterminated string literal (2)')
            k = 'abfrtvn\\"\''.find(ch)
            if k >= 0:
                uch = '\x07\x08\x0c\r\t\x0b\n\\"\''[k]
                ustr += uch
                j += 1
            elif ch == 'u':
                j, ch = self.uEscape(argstr, j + 1, startline)
                ustr += ch
            elif ch == 'U':
                j, ch = self.UEscape(argstr, j + 1, startline)
                ustr += ch
            else:
                self.BadSyntax(argstr, i, 'bad escape')
    self.BadSyntax(argstr, i, 'unterminated string literal')