from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def fix_parents(stream):
    for record in stream:
        chunks = record.get_bytes_as('chunked')
        new_key = (new_file_id, record.key[-1])
        parents = new_parents[new_key]
        yield ChunkedContentFactory(new_key, parents, record.sha1, chunks)