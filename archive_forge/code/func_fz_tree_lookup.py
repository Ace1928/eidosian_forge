from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_tree_lookup(self, key):
    """
        Class-aware wrapper for `::fz_tree_lookup()`.
        	Look for the value of a node in the tree with the given key.

        	Simple pointer equivalence is used for key.

        	Returns NULL for no match.
        """
    return _mupdf.FzTree_fz_tree_lookup(self, key)