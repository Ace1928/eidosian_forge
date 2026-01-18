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
def init_document_data(self) -> None:
    preliminary_document_data = [list(x) for x in self.config.latex_documents]
    if not preliminary_document_data:
        logger.warning(__('no "latex_documents" config value found; no documents will be written'))
        return
    self.titles: List[Tuple[str, str]] = []
    for entry in preliminary_document_data:
        docname = entry[0]
        if docname not in self.env.all_docs:
            logger.warning(__('"latex_documents" config value references unknown document %s'), docname)
            continue
        self.document_data.append(entry)
        if docname.endswith(SEP + 'index'):
            docname = docname[:-5]
        self.titles.append((docname, entry[2]))