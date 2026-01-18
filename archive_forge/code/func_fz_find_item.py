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
def fz_find_item(drop, key, type):
    """
    Class-aware wrapper for `::fz_find_item()`.
    	Find an item within the store.

    	drop: The function used to free the value (to ensure we get a
    	value of the correct type).

    	key: The key used to index the item.

    	type: Functions used to manipulate the key.

    	Returns NULL for not found, otherwise returns a pointer to the
    	value indexed by key to which a reference has been taken.
    """
    return _mupdf.fz_find_item(drop, key, type)