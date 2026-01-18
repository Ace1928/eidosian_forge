from __future__ import annotations
from sphinx.util import logging  # isort:skip
import re
import warnings
from os import getenv
from os.path import basename, dirname, join
from uuid import uuid4
from docutils import nodes
from docutils.parsers.rst.directives import choice, flag
from sphinx.errors import SphinxError
from sphinx.util import copyfile, ensuredir
from sphinx.util.display import status_iterator
from sphinx.util.nodes import set_source_info
from bokeh.document import Document
from bokeh.embed import autoload_static
from bokeh.model import Model
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .example_handler import ExampleHandler
from .util import _REPO_TOP, get_sphinx_resources
def process_source(self, source, path, js_filename):
    Model._clear_extensions()
    root, docstring = _evaluate_source(source, path, self.env)
    height_hint = root._sphinx_height_hint()
    js_path = join(self.env.bokeh_plot_auxdir, js_filename)
    js, script_tag = autoload_static(root, RESOURCES, js_filename)
    with open(js_path, 'w') as f:
        f.write(js)
    return (script_tag, js_path, source, docstring, height_hint)