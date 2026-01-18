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
def relfn2path(self, filename: str, docname: Optional[str]=None) -> Tuple[str, str]:
    """Return paths to a file referenced from a document, relative to
        documentation root and absolute.

        In the input "filename", absolute filenames are taken as relative to the
        source dir, while relative filenames are relative to the dir of the
        containing document.
        """
    filename = os_path(filename)
    if filename.startswith('/') or filename.startswith(os.sep):
        rel_fn = filename[1:]
    else:
        docdir = path.dirname(self.doc2path(docname or self.docname, base=False))
        rel_fn = path.join(docdir, filename)
    return (canon_path(path.normpath(rel_fn)), path.normpath(path.join(self.srcdir, rel_fn)))