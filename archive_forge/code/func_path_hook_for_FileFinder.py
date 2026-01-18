import _imp
import _io
import sys
import _warnings
import marshal
def path_hook_for_FileFinder(path):
    """Path hook for importlib.machinery.FileFinder."""
    if not _path_isdir(path):
        raise ImportError('only directories are supported', path=path)
    return cls(path, *loader_details)