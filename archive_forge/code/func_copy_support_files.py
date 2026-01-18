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
@progress_message(__('copying TeX support files'))
def copy_support_files(self) -> None:
    """copy TeX support files from texinputs."""
    xindy_lang_option = XINDY_LANG_OPTIONS.get(self.config.language[:2], '-L general -C utf8 ')
    xindy_cyrillic = self.config.language[:2] in XINDY_CYRILLIC_SCRIPTS
    context = {'latex_engine': self.config.latex_engine, 'xindy_use': self.config.latex_use_xindy, 'xindy_lang_option': xindy_lang_option, 'xindy_cyrillic': xindy_cyrillic}
    logger.info(bold(__('copying TeX support files...')))
    staticdirname = path.join(package_dir, 'texinputs')
    for filename in os.listdir(staticdirname):
        if not filename.startswith('.'):
            copy_asset_file(path.join(staticdirname, filename), self.outdir, context=context)
    if os.name == 'nt':
        staticdirname = path.join(package_dir, 'texinputs_win')
        copy_asset_file(path.join(staticdirname, 'Makefile_t'), self.outdir, context=context)