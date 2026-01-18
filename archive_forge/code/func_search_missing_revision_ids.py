import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
def search_missing_revision_ids(self, find_ghosts=True, revision_ids=None, if_present_ids=None, limit=None):
    """See InterRepository.search_missing_revision_ids()."""
    with self.lock_read():
        source_ids_set = self._present_source_revisions_for(revision_ids, if_present_ids)
        target_ids = set(self.target._all_possible_ids())
        possibly_present_revisions = target_ids.intersection(source_ids_set)
        actually_present_revisions = set(self.target._eliminate_revisions_not_present(possibly_present_revisions))
        required_revisions = source_ids_set.difference(actually_present_revisions)
        if revision_ids is not None:
            result_set = required_revisions
        else:
            result_set = set(self.source._eliminate_revisions_not_present(required_revisions))
        if limit is not None:
            topo_ordered = self.source.get_graph().iter_topo_order(result_set)
            result_set = set(itertools.islice(topo_ordered, limit))
        return self.source.revision_ids_to_search_result(result_set)