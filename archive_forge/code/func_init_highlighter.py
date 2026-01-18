import html
import os
import posixpath
import re
import sys
import warnings
from datetime import datetime
from os import path
from typing import IO, Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type
from urllib.parse import quote
import docutils.readers.doctree
from docutils import nodes
from docutils.core import Publisher
from docutils.frontend import OptionParser
from docutils.io import DocTreeInput, StringOutput
from docutils.nodes import Node
from docutils.utils import relative_path
from sphinx import __display_version__, package_dir
from sphinx import version_info as sphinx_version
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx70Warning, deprecated_alias
from sphinx.domains import Domain, Index, IndexEntry
from sphinx.environment import BuildEnvironment
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import ConfigError, ThemeError
from sphinx.highlighting import PygmentsBridge
from sphinx.locale import _, __
from sphinx.search import js_index
from sphinx.theming import HTMLThemeFactory
from sphinx.util import isurl, logging, md5, progress_message, status_iterator
from sphinx.util.docutils import new_document
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import format_date
from sphinx.util.inventory import InventoryFile
from sphinx.util.matching import DOTFILES, Matcher, patmatch
from sphinx.util.osutil import copyfile, ensuredir, os_path, relative_uri
from sphinx.util.tags import Tags
from sphinx.writers.html import HTMLTranslator, HTMLWriter
from sphinx.writers.html5 import HTML5Translator
import sphinxcontrib.serializinghtml  # NOQA
import sphinx.builders.dirhtml  # NOQA
import sphinx.builders.singlehtml  # NOQA
def init_highlighter(self) -> None:
    if self.config.pygments_style is not None:
        style = self.config.pygments_style
    elif self.theme:
        style = self.theme.get_config('theme', 'pygments_style', 'none')
    else:
        style = 'sphinx'
    self.highlighter = PygmentsBridge('html', style)
    if self.theme:
        dark_style = self.theme.get_config('theme', 'pygments_dark_style', None)
    else:
        dark_style = None
    if dark_style is not None:
        self.dark_highlighter = PygmentsBridge('html', dark_style)
        self.app.add_css_file('pygments_dark.css', media='(prefers-color-scheme: dark)', id='pygments_dark_css')
    else:
        self.dark_highlighter = None