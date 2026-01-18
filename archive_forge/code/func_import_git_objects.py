import posixpath
import stat
from dulwich.object_store import tree_lookup_path
from dulwich.objects import (S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Commit, Tag,
from .. import debug, errors, osutils, trace
from ..bzr.inventory import (InventoryDirectory, InventoryFile, InventoryLink,
from ..bzr.inventorytree import InventoryRevisionTree
from ..bzr.testament import StrictTestament3
from ..bzr.versionedfile import ChunkedContentFactory
from ..errors import BzrError
from ..revision import NULL_REVISION
from ..transport import NoSuchFile
from ..tree import InterTree
from ..tsort import topo_sort
from .mapping import (DEFAULT_FILE_MODE, decode_git_path, mode_is_executable,
from .object_store import LRUTreeCache, _tree_to_objects
def import_git_objects(repo, mapping, object_iter, target_git_object_retriever, heads, pb=None, limit=None):
    """Import a set of git objects into a bzr repository.

    :param repo: Target Bazaar repository
    :param mapping: Mapping to use
    :param object_iter: Iterator over Git objects.
    :return: Tuple with pack hints and last imported revision id
    """

    def lookup_object(sha):
        try:
            return object_iter[sha]
        except KeyError:
            return target_git_object_retriever[sha]
    graph = []
    checked = set()
    heads = list(set(heads))
    trees_cache = LRUTreeCache(repo)
    while heads:
        if pb is not None:
            pb.update('finding revisions to fetch', len(graph), None)
        head = heads.pop()
        if head == ZERO_SHA:
            continue
        if not isinstance(head, bytes):
            raise TypeError(head)
        try:
            o = lookup_object(head)
        except KeyError:
            continue
        if isinstance(o, Commit):
            rev, roundtrip_revid, verifiers = mapping.import_commit(o, mapping.revision_id_foreign_to_bzr, strict=True)
            if repo.has_revision(rev.revision_id) or (roundtrip_revid and repo.has_revision(roundtrip_revid)):
                continue
            graph.append((o.id, o.parents))
            heads.extend([p for p in o.parents if p not in checked])
        elif isinstance(o, Tag):
            if o.object[1] not in checked:
                heads.append(o.object[1])
        else:
            trace.warning('Unable to import head object %r' % o)
        checked.add(o.id)
    del checked
    batch_size = 1000
    revision_ids = topo_sort(graph)
    pack_hints = []
    if limit is not None:
        revision_ids = revision_ids[:limit]
    last_imported = None
    for offset in range(0, len(revision_ids), batch_size):
        target_git_object_retriever.start_write_group()
        try:
            repo.start_write_group()
            try:
                for i, head in enumerate(revision_ids[offset:offset + batch_size]):
                    if pb is not None:
                        pb.update('fetching revisions', offset + i, len(revision_ids))
                    import_git_commit(repo, mapping, head, lookup_object, target_git_object_retriever, trees_cache, strict=True)
                    last_imported = head
            except BaseException:
                repo.abort_write_group()
                raise
            else:
                hint = repo.commit_write_group()
                if hint is not None:
                    pack_hints.extend(hint)
        except BaseException:
            target_git_object_retriever.abort_write_group()
            raise
        else:
            target_git_object_retriever.commit_write_group()
    return (pack_hints, last_imported)