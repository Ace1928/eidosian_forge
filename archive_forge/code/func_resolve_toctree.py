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
def resolve_toctree(self, docname: str, builder: 'Builder', toctree: addnodes.toctree, prune: bool=True, maxdepth: int=0, titles_only: bool=False, collapse: bool=False, includehidden: bool=False) -> Optional[Node]:
    """Resolve a *toctree* node into individual bullet lists with titles
        as items, returning None (if no containing titles are found) or
        a new node.

        If *prune* is True, the tree is pruned to *maxdepth*, or if that is 0,
        to the value of the *maxdepth* option on the *toctree* node.
        If *titles_only* is True, only toplevel document titles will be in the
        resulting tree.
        If *collapse* is True, all branches not containing docname will
        be collapsed.
        """
    return TocTree(self).resolve(docname, builder, toctree, prune, maxdepth, titles_only, collapse, includehidden)