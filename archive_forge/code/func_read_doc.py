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
def read_doc(self, docname: str) -> None:
    """Parse a file and add/update inventory entries for the doctree."""
    self.env.prepare_settings(docname)
    docutilsconf = path.join(self.confdir, 'docutils.conf')
    if path.isfile(docutilsconf):
        self.env.note_dependency(docutilsconf)
    filename = self.env.doc2path(docname)
    filetype = get_filetype(self.app.config.source_suffix, filename)
    publisher = self.app.registry.get_publisher(self.app, filetype)
    publisher.settings.record_dependencies = DependencyList()
    with sphinx_domains(self.env), rst.default_role(docname, self.config.default_role):
        codecs.register_error('sphinx', UnicodeDecodeErrorHandler(docname))
        publisher.set_source(source_path=filename)
        publisher.publish()
        doctree = publisher.document
    self.env.all_docs[docname] = max(time.time(), path.getmtime(self.env.doc2path(docname)))
    self.env.temp_data.clear()
    self.env.ref_context.clear()
    self.write_doctree(docname, doctree)