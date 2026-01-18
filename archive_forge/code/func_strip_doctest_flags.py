import sys
from typing import Any, Dict, List, NamedTuple
from docutils import nodes
from docutils.nodes import Node, TextElement
from pygments.lexers import PythonConsoleLexer, guess_lexer
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import doctest
from sphinx.transforms import SphinxTransform
def strip_doctest_flags(self, node: TextElement) -> None:
    if not node.get('trim_flags', self.config.trim_doctest_flags):
        return
    source = node.rawsource
    source = doctest.blankline_re.sub('', source)
    source = doctest.doctestopt_re.sub('', source)
    node.rawsource = source
    node[:] = [nodes.Text(source)]