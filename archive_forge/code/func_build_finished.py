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
def build_finished(app, exception):
    files = sorted(app.env.bokeh_plot_files)
    files_iter = status_iterator(files, 'copying bokeh-plot files... ', 'brown', len(files), app.verbosity, stringify_func=lambda x: basename(x[0]))
    for file, docpath in files_iter:
        target = join(app.builder.outdir, docpath, basename(file))
        ensuredir(dirname(target))
        try:
            copyfile(file, target)
        except OSError as e:
            raise SphinxError(f'cannot copy local file {file!r}, reason: {e}')