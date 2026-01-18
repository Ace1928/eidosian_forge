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
def ll_fz_tree_insert(root, key, value):
    """
    Low-level wrapper for `::fz_tree_insert()`.
    Insert a new key/value pair and rebalance the tree.
    Return the new root of the tree after inserting and rebalancing.
    May be called with a NULL root to create a new tree.

    No data is copied into the tree structure; key and value are
    merely kept as pointers.
    """
    return _mupdf.ll_fz_tree_insert(root, key, value)