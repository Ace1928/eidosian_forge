import contextlib
from importlib import import_module
import os
import sys
from . import _util
def iter_project_files(project, relative=False, **kwargs):
    """Yield (dirname, basename, filename) for all files in the project."""
    if relative:
        with _util.cwd(VENDORED_ROOT):
            for result in _util.iter_all_files(project, **kwargs):
                yield result
    else:
        root = project_root(project)
        for result in _util.iter_all_files(root, **kwargs):
            yield result