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
def iter_files_bytes(self, desired_files):
    """Iterate through file versions.

        Files will not necessarily be returned in the order they occur in
        desired_files.  No specific order is guaranteed.

        Yields pairs of identifier, bytes_iterator.  identifier is an opaque
        value supplied by the caller as part of desired_files.  It should
        uniquely identify the file version in the caller's context.  (Examples:
        an index number or a TreeTransform trans_id.)

        bytes_iterator is an iterable of bytestrings for the file.  The
        kind of iterable and length of the bytestrings are unspecified, but for
        this implementation, it is a list of bytes produced by
        VersionedFile.get_record_stream().

        :param desired_files: a list of (file_id, revision_id, identifier)
            triples
        """
    per_revision = {}
    for file_id, revision_id, identifier in desired_files:
        per_revision.setdefault(revision_id, []).append((file_id, identifier))
    for revid, files in per_revision.items():
        try:
            commit_id, mapping = self.lookup_bzr_revision_id(revid)
        except errors.NoSuchRevision:
            raise errors.RevisionNotPresent(revid, self)
        try:
            commit = self._git.object_store[commit_id]
        except KeyError:
            raise errors.RevisionNotPresent(revid, self)
        root_tree = commit.tree
        for fileid, identifier in files:
            try:
                path = mapping.parse_file_id(fileid)
            except ValueError:
                raise errors.RevisionNotPresent((fileid, revid), self)
            try:
                mode, item_id = tree_lookup_path(self._git.object_store.__getitem__, root_tree, encode_git_path(path))
                obj = self._git.object_store[item_id]
            except KeyError:
                raise errors.RevisionNotPresent((fileid, revid), self)
            else:
                if obj.type_name == b'tree':
                    yield (identifier, [])
                elif obj.type_name == b'blob':
                    yield (identifier, obj.chunked)
                else:
                    raise AssertionError('file text resolved to %r' % obj)