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
def get_outdated_docs(self) -> Iterator[str]:
    try:
        with open(path.join(self.outdir, '.buildinfo'), encoding='utf-8') as fp:
            buildinfo = BuildInfo.load(fp)
        if self.build_info != buildinfo:
            logger.debug('[build target] did not match: build_info ')
            yield from self.env.found_docs
            return
    except ValueError as exc:
        logger.warning(__('Failed to read build info file: %r'), exc)
    except OSError:
        pass
    if self.templates:
        template_mtime = self.templates.newest_template_mtime()
    else:
        template_mtime = 0
    for docname in self.env.found_docs:
        if docname not in self.env.all_docs:
            logger.debug('[build target] did not in env: %r', docname)
            yield docname
            continue
        targetname = self.get_outfilename(docname)
        try:
            targetmtime = path.getmtime(targetname)
        except Exception:
            targetmtime = 0
        try:
            srcmtime = max(path.getmtime(self.env.doc2path(docname)), template_mtime)
            if srcmtime > targetmtime:
                logger.debug('[build target] targetname %r(%s), template(%s), docname %r(%s)', targetname, datetime.utcfromtimestamp(targetmtime), datetime.utcfromtimestamp(template_mtime), docname, datetime.utcfromtimestamp(path.getmtime(self.env.doc2path(docname))))
                yield docname
        except OSError:
            pass