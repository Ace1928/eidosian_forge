import contextlib
import os
def iter_tree(root, prune_dir=None, exclude_file=None):
    """Yield (dirname, files) for each directory in the tree.

    The list of files is actually a list of (basename, filename).

    This is an alternative to os.walk() with filtering."""
    pending = [root]
    while pending:
        dirname = pending.pop(0)
        files = []
        for _, b, f in _iter_files(dirname, pending, prune_dir, exclude_file):
            files.append((b, f))
        yield (dirname, files)