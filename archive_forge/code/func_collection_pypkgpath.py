from __future__ import (absolute_import, division, print_function)
import os
def collection_pypkgpath(self):
    """Configure the Python package path so that pytest can find our collections."""
    for parent in self.parts(reverse=True):
        if str(parent) == ANSIBLE_COLLECTIONS_PATH:
            return parent
    raise Exception('File "%s" not found in collection path "%s".' % (self.strpath, ANSIBLE_COLLECTIONS_PATH))