from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
def merge_xrefs(app, env, docnames, other):
    if not hasattr(env, 'all_sampledata_xrefs'):
        env.all_sampledata_xrefs = []
    if hasattr(other, 'all_sampledata_xrefs'):
        env.all_sampledata_xrefs.extend(other.all_sampledata_xrefs)