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
def fz_store_item(key, val, itemsize, type):
    """
    Class-aware wrapper for `::fz_store_item()`.
    	Add an item to the store.

    	Add an item into the store, returning NULL for success. If an
    	item with the same key is found in the store, then our item will
    	not be inserted, and the function will return a pointer to that
    	value instead. This function takes its own reference to val, as
    	required (i.e. the caller maintains ownership of its own
    	reference).

    	key: The key used to index the item.

    	val: The value to store.

    	itemsize: The size in bytes of the value (as counted towards the
    	store size).

    	type: Functions used to manipulate the key.
    """
    return _mupdf.fz_store_item(key, val, itemsize, type)