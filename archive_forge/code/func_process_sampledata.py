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
def process_sampledata(self, source):
    if not hasattr(self.env, 'solved_sampledata'):
        self.env.solved_sampledata = []
    file, lineno = self.get_source_info()
    if '/docs/examples/' in file and file not in self.env.solved_sampledata:
        self.env.solved_sampledata.append(file)
        if not hasattr(self.env, 'all_sampledata_xrefs'):
            self.env.all_sampledata_xrefs = []
        if not hasattr(self.env, 'all_gallery_overview'):
            self.env.all_gallery_overview = []
        self.env.all_gallery_overview.append({'docname': self.env.docname})
        regex = '(:|bokeh\\.)sampledata(:|\\.| import )\\s*(\\w+(\\,\\s*\\w+)*)'
        matches = re.findall(regex, source)
        if matches:
            keywords = set()
            for m in matches:
                keywords.update(m[2].replace(' ', '').split(','))
            for keyword in keywords:
                self.env.all_sampledata_xrefs.append({'docname': self.env.docname, 'keyword': keyword})