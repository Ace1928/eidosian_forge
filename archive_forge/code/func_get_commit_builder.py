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
def get_commit_builder(self, branch, parents, config, timestamp=None, timezone=None, committer=None, revprops=None, revision_id=None, lossy=False):
    self._check_ascii_revisionid(revision_id, self.get_commit_builder)
    result = VersionedFileCommitBuilder(self, parents, config, timestamp, timezone, committer, revprops, revision_id, lossy=lossy)
    self.start_write_group()
    return result