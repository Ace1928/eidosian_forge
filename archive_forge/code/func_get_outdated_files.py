import os
import pickle
import warnings
from collections import defaultdict
from copy import copy
from datetime import datetime
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, Optional,
import docutils
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import BuildEnvironmentError, DocumentError, ExtensionError, SphinxError
from sphinx.events import EventManager
from sphinx.locale import __
from sphinx.project import Project
from sphinx.transforms import SphinxTransformer
from sphinx.util import DownloadFiles, FilenameUniqDict, logging
from sphinx.util.docutils import LoggingReporter
from sphinx.util.i18n import CatalogRepository, docname_to_domain
from sphinx.util.nodes import is_translatable
from sphinx.util.osutil import canon_path, os_path
def get_outdated_files(self, config_changed: bool) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (added, changed, removed) sets."""
    removed = set(self.all_docs) - self.found_docs
    added: Set[str] = set()
    changed: Set[str] = set()
    if config_changed:
        added = self.found_docs
    else:
        for docname in self.found_docs:
            if docname not in self.all_docs:
                logger.debug('[build target] added %r', docname)
                added.add(docname)
                continue
            filename = path.join(self.doctreedir, docname + '.doctree')
            if not path.isfile(filename):
                logger.debug('[build target] changed %r', docname)
                changed.add(docname)
                continue
            if docname in self.reread_always:
                logger.debug('[build target] changed %r', docname)
                changed.add(docname)
                continue
            mtime = self.all_docs[docname]
            newmtime = path.getmtime(self.doc2path(docname))
            if newmtime > mtime:
                logger.debug('[build target] outdated %r: %s -> %s', docname, datetime.utcfromtimestamp(mtime), datetime.utcfromtimestamp(newmtime))
                changed.add(docname)
                continue
            for dep in self.dependencies[docname]:
                try:
                    deppath = path.join(self.srcdir, dep)
                    if not path.isfile(deppath):
                        changed.add(docname)
                        break
                    depmtime = path.getmtime(deppath)
                    if depmtime > mtime:
                        changed.add(docname)
                        break
                except OSError:
                    changed.add(docname)
                    break
    return (added, changed, removed)