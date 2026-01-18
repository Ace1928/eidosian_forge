from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
def process_gallery_overview(app, doctree, fromdocname):
    env = app.builder.env
    if not hasattr(env, 'all_gallery_overview'):
        env.all_gallery_overview = []
    for node in doctree.traverse(gallery_xrefs):
        ref_dict = {}
        for s in env.all_gallery_overview:
            sp = s['docname'].split('/')
            if node.subfolder == 'all' or sp[-2] == node.subfolder:
                letter = sp[-1][0].upper()
                if letter in ref_dict and s not in ref_dict[letter]:
                    ref_dict[letter].append(s)
                else:
                    ref_dict[letter] = [s]
        content = []
        for letter, refs in sorted(ref_dict.items()):
            para = nodes.paragraph()
            para += nodes.rubric(_(letter), _(letter))
            for ref in sort_by_basename(refs):
                docname = ref['docname']
                ref_name = basename(docname)
                if node.subfolder == 'all':
                    ref_name += f' ({docname.split('/')[-2]})'
                para += add_bullet_point(app, fromdocname, docname, ref_name)
            content.append(para)
        node.replace_self(content)