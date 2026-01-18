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
def pyobject_filter(self, lines: List[str], location: Optional[Tuple[str, int]]=None) -> List[str]:
    pyobject = self.options.get('pyobject')
    if pyobject:
        from sphinx.pycode import ModuleAnalyzer
        analyzer = ModuleAnalyzer.for_file(self.filename, '')
        tags = analyzer.find_tags()
        if pyobject not in tags:
            raise ValueError(__('Object named %r not found in include file %r') % (pyobject, self.filename))
        else:
            start = tags[pyobject][1]
            end = tags[pyobject][2]
            lines = lines[start - 1:end]
            if 'lineno-match' in self.options:
                self.lineno_start = start
    return lines