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
def get_code_language(self):
    """
        This is largely copied from bokeh.sphinxext.bokeh_plot.run
        """
    js_source = self.get_js_source()
    if self.options.get('include_html', False):
        resources = get_sphinx_resources(include_bokehjs_api=True)
        html_source = BJS_HTML.render(css_files=resources.css_files, js_files=resources.js_files, hashes=resources.hashes, bjs_script=js_source)
        return [html_source, 'html']
    else:
        return [js_source, 'javascript']