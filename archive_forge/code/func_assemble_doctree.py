import os
import warnings
from os import path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from docutils.frontend import OptionParser
from docutils.nodes import Node
import sphinx.builders.latex.nodes  # NOQA  # Workaround: import this before writer to avoid ImportError
from sphinx import addnodes, highlighting, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.builders.latex.constants import ADDITIONAL_SETTINGS, DEFAULT_SETTINGS, SHORTHANDOFF
from sphinx.builders.latex.theming import Theme, ThemeFactory
from sphinx.builders.latex.util import ExtBabel
from sphinx.config import ENUM, Config
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import NoUri, SphinxError
from sphinx.locale import _, __
from sphinx.util import logging, progress_message, status_iterator, texescape
from sphinx.util.console import bold, darkgreen  # type: ignore
from sphinx.util.docutils import SphinxFileOutput, new_document
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.nodes import inline_all_toctrees
from sphinx.util.osutil import SEP, make_filename_from_project
from sphinx.util.template import LaTeXRenderer
from sphinx.writers.latex import LaTeXTranslator, LaTeXWriter
from docutils import nodes  # isort:skip
def assemble_doctree(self, indexfile: str, toctree_only: bool, appendices: List[str]) -> nodes.document:
    self.docnames = set([indexfile] + appendices)
    logger.info(darkgreen(indexfile) + ' ', nonl=True)
    tree = self.env.get_doctree(indexfile)
    tree['docname'] = indexfile
    if toctree_only:
        new_tree = new_document('<latex output>')
        new_sect = nodes.section()
        new_sect += nodes.title('<Set title in conf.py>', '<Set title in conf.py>')
        new_tree += new_sect
        for node in tree.findall(addnodes.toctree):
            new_sect += node
        tree = new_tree
    largetree = inline_all_toctrees(self, self.docnames, indexfile, tree, darkgreen, [indexfile])
    largetree['docname'] = indexfile
    for docname in appendices:
        appendix = self.env.get_doctree(docname)
        appendix['docname'] = docname
        largetree.append(appendix)
    logger.info('')
    logger.info(__('resolving references...'))
    self.env.resolve_references(largetree, indexfile, self)
    for pendingnode in largetree.findall(addnodes.pending_xref):
        docname = pendingnode['refdocname']
        sectname = pendingnode['refsectname']
        newnodes: List[Node] = [nodes.emphasis(sectname, sectname)]
        for subdir, title in self.titles:
            if docname.startswith(subdir):
                newnodes.append(nodes.Text(_(' (in ')))
                newnodes.append(nodes.emphasis(title, title))
                newnodes.append(nodes.Text(')'))
                break
        else:
            pass
        pendingnode.replace_self(newnodes)
    return largetree