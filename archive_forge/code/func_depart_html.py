from __future__ import annotations
import logging  # isort:skip
from os.path import basename, join
from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from sphinx.directives.code import CodeBlock, container_wrapper, dedent_lines
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging, parselinenos
from sphinx.util.nodes import set_source_info
from . import PARALLEL_SAFE
from .templates import (
from .util import get_sphinx_resources
@staticmethod
def depart_html(visitor, node):
    visitor.body.append(BJS_EPILOGUE.render(title=node['title'], enable_codepen=not node['disable_codepen'], js_source=node['js_source']))