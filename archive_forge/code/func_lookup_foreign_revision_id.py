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
def lookup_foreign_revision_id(self, foreign_revid, mapping=None):
    """Lookup a revision id.

        :param foreign_revid: Foreign revision id to look up
        :param mapping: Mapping to use (use default mapping if not specified)
        :raise KeyError: If foreign revision was not found
        :return: bzr revision id
        """
    if not isinstance(foreign_revid, bytes):
        raise TypeError(foreign_revid)
    if mapping is None:
        mapping = self.get_mapping()
    if foreign_revid == ZERO_SHA:
        return _mod_revision.NULL_REVISION
    unpeeled, peeled = peel_sha(self._git.object_store, foreign_revid)
    if not isinstance(peeled, Commit):
        raise NotCommitError(peeled.id)
    revid = mapping.get_revision_id(peeled)
    return revid