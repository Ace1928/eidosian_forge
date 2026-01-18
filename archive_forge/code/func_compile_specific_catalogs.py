import codecs
import pickle
import time
import warnings
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import DependencyList
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import CONFIG_CHANGED_REASON, CONFIG_OK, BuildEnvironment
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import SphinxError
from sphinx.events import EventManager
from sphinx.locale import __
from sphinx.util import (UnicodeDecodeErrorHandler, get_filetype, import_object, logging,
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import sphinx_domains
from sphinx.util.i18n import CatalogInfo, CatalogRepository, docname_to_domain
from sphinx.util.osutil import SEP, ensuredir, relative_uri, relpath
from sphinx.util.parallel import ParallelTasks, SerialTasks, make_chunks, parallel_available
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType
from sphinx import directives  # NOQA isort:skip
from sphinx import roles  # NOQA isort:skip
def compile_specific_catalogs(self, specified_files: List[str]) -> None:

    def to_domain(fpath: str) -> Optional[str]:
        docname = self.env.path2doc(path.abspath(fpath))
        if docname:
            return docname_to_domain(docname, self.config.gettext_compact)
        else:
            return None
    catalogs = set()
    domains = set(map(to_domain, specified_files))
    repo = CatalogRepository(self.srcdir, self.config.locale_dirs, self.config.language, self.config.source_encoding)
    for catalog in repo.catalogs:
        if catalog.domain in domains and catalog.is_outdated():
            catalogs.add(catalog)
    message = __('targets for %d po files that are specified') % len(catalogs)
    self.compile_catalogs(catalogs, message)