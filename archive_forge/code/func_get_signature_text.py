from io import BytesIO
from dulwich.errors import NotCommitError
from dulwich.object_store import peel_sha, tree_lookup_path
from dulwich.objects import ZERO_SHA, Commit
from .. import check, errors
from .. import graph as _mod_graph
from .. import lock, repository
from .. import revision as _mod_revision
from .. import trace, transactions, ui
from ..decorators import only_raises
from ..foreign import ForeignRepository
from .filegraph import GitFileLastChangeScanner, GitFileParentProvider
from .mapping import (default_mapping, encode_git_path, foreign_vcs_git,
from .tree import GitRevisionTree
def get_signature_text(self, revision_id):
    git_commit_id, mapping = self.lookup_bzr_revision_id(revision_id)
    try:
        commit = self._git.object_store[git_commit_id]
    except KeyError:
        raise errors.NoSuchRevision(self, revision_id)
    if commit.gpgsig is None:
        raise errors.NoSuchRevision(self, revision_id)
    return commit.gpgsig