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
def apply_post_transforms(self, doctree: nodes.document, docname: str) -> None:
    """Apply all post-transforms."""
    try:
        backup = copy(self.temp_data)
        self.temp_data['docname'] = docname
        transformer = SphinxTransformer(doctree)
        transformer.set_environment(self)
        transformer.add_transforms(self.app.registry.get_post_transforms())
        transformer.apply_transforms()
    finally:
        self.temp_data = backup
    self.events.emit('doctree-resolved', doctree, docname)