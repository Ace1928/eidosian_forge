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
def fz_store_scavenge(size, phase):
    """
    Class-aware wrapper for `::fz_store_scavenge()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_store_scavenge(size_t size)` => `(int, int phase)`

    	Internal function used as part of the scavenging
    	allocator; when we fail to allocate memory, before returning a
    	failure to the caller, we try to scavenge space within the store
    	by evicting at least 'size' bytes. The allocator then retries.

    	size: The number of bytes we are trying to have free.

    	phase: What phase of the scavenge we are in. Updated on exit.

    	Returns non zero if we managed to free any memory.
    """
    return _mupdf.fz_store_scavenge(size, phase)