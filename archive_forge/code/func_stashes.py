import os
from .file import GitFile
from .index import commit_tree, iter_fresh_objects
from .reflog import drop_reflog_entry, read_reflog
def stashes(self):
    try:
        with GitFile(self._reflog_path, 'rb') as f:
            return reversed(list(read_reflog(f)))
    except FileNotFoundError:
        return []