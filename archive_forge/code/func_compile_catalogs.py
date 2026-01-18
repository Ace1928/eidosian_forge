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
def compile_catalogs(self, catalogs: Set[CatalogInfo], message: str) -> None:
    if not self.config.gettext_auto_build:
        return

    def cat2relpath(cat: CatalogInfo) -> str:
        return relpath(cat.mo_path, self.env.srcdir).replace(path.sep, SEP)
    logger.info(bold(__('building [mo]: ')) + message)
    for catalog in status_iterator(catalogs, __('writing output... '), 'darkgreen', len(catalogs), self.app.verbosity, stringify_func=cat2relpath):
        catalog.write_mo(self.config.language, self.config.gettext_allow_fuzzy_translations)