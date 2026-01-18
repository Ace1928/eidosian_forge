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
def init_multilingual(self) -> None:
    if self.context['latex_engine'] == 'pdflatex':
        if not self.babel.uses_cyrillic():
            if 'X2' in self.context['fontenc']:
                self.context['substitutefont'] = '\\usepackage{substitutefont}'
                self.context['textcyrillic'] = '\\usepackage[Xtwo]{sphinxpackagecyrillic}'
            elif 'T2A' in self.context['fontenc']:
                self.context['substitutefont'] = '\\usepackage{substitutefont}'
                self.context['textcyrillic'] = '\\usepackage[TtwoA]{sphinxpackagecyrillic}'
        if 'LGR' in self.context['fontenc']:
            self.context['substitutefont'] = '\\usepackage{substitutefont}'
        else:
            self.context['textgreek'] = ''
        if self.context['substitutefont'] == '':
            self.context['fontsubstitution'] = ''
    if self.context['babel']:
        self.context['classoptions'] += ',' + self.babel.get_language()
        self.context['multilingual'] = self.context['babel']
        self.context['shorthandoff'] = SHORTHANDOFF
        if self.babel.uses_cyrillic() and 'fontpkg' not in self.config.latex_elements:
            self.context['fontpkg'] = ''
    elif self.context['polyglossia']:
        self.context['classoptions'] += ',' + self.babel.get_language()
        options = self.babel.get_mainlanguage_options()
        if options:
            language = '\\setmainlanguage[%s]{%s}' % (options, self.babel.get_language())
        else:
            language = '\\setmainlanguage{%s}' % self.babel.get_language()
        self.context['multilingual'] = '%s\n%s' % (self.context['polyglossia'], language)