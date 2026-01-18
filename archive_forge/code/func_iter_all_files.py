import contextlib
import os
def iter_all_files(root, prune_dir=None, exclude_file=None):
    """Yield (dirname, basename, filename) for each file in the tree.

    This is an alternative to os.walk() that flattens out the tree and
    with filtering.
    """
    pending = [root]
    while pending:
        dirname = pending.pop(0)
        for result in _iter_files(dirname, pending, prune_dir, exclude_file):
            yield result