from __future__ import (absolute_import, division, print_function)
import os
def collection_resolve_package_path(path):
    """Configure the Python package path so that pytest can find our collections."""
    for parent in path.parents:
        if str(parent) == ANSIBLE_COLLECTIONS_PATH:
            return parent
    raise Exception('File "%s" not found in collection path "%s".' % (path, ANSIBLE_COLLECTIONS_PATH))