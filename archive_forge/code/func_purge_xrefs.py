from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
def purge_xrefs(app, env, docname):
    if not hasattr(env, 'all_sampledata_xrefs'):
        return
    env.all_sampledata_xrefs = [xref for xref in env.all_sampledata_xrefs if xref['docname'] != docname]