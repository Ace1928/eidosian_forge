from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
def process_sampledata_xrefs(app, doctree, fromdocname):
    env = app.builder.env
    if not hasattr(env, 'all_sampledata_xrefs'):
        env.all_sampledata_xrefs = []
    for node in doctree.traverse(sampledata_list):
        refs = []
        for s in env.all_sampledata_xrefs:
            if s['keyword'] == node.sampledata_key and s not in refs:
                refs.append(s)
        content = []
        if refs:
            list_ref_names = []
            para = nodes.paragraph()
            para += nodes.rubric('Examples', 'Examples')
            for ref in sort_by_basename(refs):
                ref_name = ref['basename']
                if ref_name in list_ref_names:
                    ref_name += f' ({ref['docname'].split('/')[-2]})'
                list_ref_names.append(ref_name)
                para += add_bullet_point(app, fromdocname, ref['docname'], ref_name)
            content.append(para)
        node.replace_self(content)