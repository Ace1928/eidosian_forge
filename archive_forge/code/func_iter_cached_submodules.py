from typing import Iterator, Tuple
from .object_store import iter_tree_contents
from .objects import S_ISGITLINK
def iter_cached_submodules(store, root_tree_id: bytes) -> Iterator[Tuple[str, bytes]]:
    """Iterate over cached submodules.

    Args:
      store: Object store to iterate
      root_tree_id: SHA of root tree

    Returns:
      Iterator over over (path, sha) tuples
    """
    for entry in iter_tree_contents(store, root_tree_id):
        if S_ISGITLINK(entry.mode):
            yield (entry.path, entry.sha)