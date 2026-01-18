from __future__ import annotations
import html
import os
import posixpath
import re
from collections.abc import Iterable
from os import path
from typing import Any, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import canon_path, make_filename
from sphinx.util.template import SphinxRenderer
def get_project_files(self, outdir: str | os.PathLike[str]) -> list[str]:
    project_files = []
    staticdir = path.join(outdir, '_static')
    imagesdir = path.join(outdir, self.imagedir)
    for root, dirs, files in os.walk(outdir):
        resourcedir = root.startswith((staticdir, imagesdir))
        for fn in sorted(files):
            if resourcedir and (not fn.endswith('.js')) or fn.endswith('.html'):
                filename = path.relpath(path.join(root, fn), outdir)
                project_files.append(canon_path(filename))
    return project_files