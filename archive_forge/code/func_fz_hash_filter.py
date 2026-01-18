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
def fz_hash_filter(self, state, callback):
    """
        Class-aware wrapper for `::fz_hash_filter()`.
        	Iterate over the entries in a hash table, removing all the ones where callback returns true.
        	Does NOT free the value of the entry, so the caller is expected to take care of this.
        """
    return _mupdf.FzHashTable_fz_hash_filter(self, state, callback)