from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
def sort_by_basename(refs):
    refs = [{'basename': basename(ref['docname']), 'docname': ref['docname']} for ref in refs]
    sorted_refs = []
    for key in sorted([basename(ref['basename']) for ref in refs]):
        for i, value in enumerate(refs):
            if key == value['basename']:
                sorted_refs.append(refs.pop(i))
    return sorted_refs