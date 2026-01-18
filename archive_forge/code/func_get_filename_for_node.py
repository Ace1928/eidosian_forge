import doctest
import re
import sys
import time
from io import StringIO
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement
from docutils.parsers.rst import directives
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
import sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relpath
from sphinx.util.typing import OptionSpec
def get_filename_for_node(self, node: Node, docname: str) -> str:
    """Try to get the file which actually contains the doctest, not the
        filename of the document it's included in."""
    try:
        filename = relpath(node.source, self.env.srcdir).rsplit(':docstring of ', maxsplit=1)[0]
    except Exception:
        filename = self.env.doc2path(docname, False)
    return filename