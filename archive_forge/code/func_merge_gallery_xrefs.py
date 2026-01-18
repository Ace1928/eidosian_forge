from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
def merge_gallery_xrefs(app, env, docnames, other):
    if not hasattr(env, 'all_gallery_overview'):
        env.all_gallery_overview = []
    if hasattr(other, 'all_gallery_overview'):
        env.all_gallery_overview.extend(other.all_gallery_overview)