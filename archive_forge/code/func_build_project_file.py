from __future__ import annotations
import html
import os
from os import path
from typing import Any
from docutils import nodes
from docutils.nodes import Element, Node, document
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.config import Config
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import make_filename_from_project, relpath
from sphinx.util.template import SphinxRenderer
@progress_message(__('writing project file'))
def build_project_file(self) -> None:
    """Create a project file (.hhp) on outdir."""
    project_files: list[str] = []
    for root, dirs, files in os.walk(self.outdir):
        dirs.sort()
        files.sort()
        in_staticdir = root.startswith(path.join(self.outdir, '_static'))
        for fn in sorted(files):
            if in_staticdir and (not fn.endswith('.js')) or fn.endswith('.html'):
                fn = relpath(path.join(root, fn), self.outdir)
                project_files.append(fn.replace(os.sep, '\\'))
    filename = path.join(self.outdir, self.config.htmlhelp_basename + '.hhp')
    with open(filename, 'w', encoding=self.encoding, errors='xmlcharrefreplace') as f:
        context = {'outname': self.config.htmlhelp_basename, 'title': self.config.html_title, 'version': self.config.version, 'project': self.config.project, 'lcid': self.lcid, 'master_doc': self.config.master_doc + self.out_suffix, 'files': project_files}
        body = self.render('project.hhp', context)
        f.write(body)