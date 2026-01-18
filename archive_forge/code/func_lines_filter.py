import sys
import textwrap
from difflib import unified_diff
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.config import Config
from sphinx.directives import optional_int
from sphinx.locale import __
from sphinx.util import logging, parselinenos
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
def lines_filter(self, lines: List[str], location: Optional[Tuple[str, int]]=None) -> List[str]:
    linespec = self.options.get('lines')
    if linespec:
        linelist = parselinenos(linespec, len(lines))
        if any((i >= len(lines) for i in linelist)):
            logger.warning(__('line number spec is out of range(1-%d): %r') % (len(lines), linespec), location=location)
        if 'lineno-match' in self.options:
            first = linelist[0]
            if all((first + i == n for i, n in enumerate(linelist))):
                self.lineno_start += linelist[0]
            else:
                raise ValueError(__('Cannot use "lineno-match" with a disjoint set of "lines"'))
        lines = [lines[n] for n in linelist if n < len(lines)]
        if lines == []:
            raise ValueError(__('Line spec %r: no lines pulled from include file %r') % (linespec, self.filename))
    return lines