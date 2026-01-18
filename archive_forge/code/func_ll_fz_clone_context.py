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
def ll_fz_clone_context():
    """
    Low-level wrapper for `::fz_clone_context()`.
    Make a clone of an existing context.

    This function is meant to be used in multi-threaded
    applications where each thread requires its own context, yet
    parts of the global state, for example caching, are shared.

    ctx: Context obtained from fz_new_context to make a copy of.
    ctx must have had locks and lock/functions setup when created.
    The two contexts will share the memory allocator, resource
    store, locks and lock/unlock functions. They will each have
    their own exception stacks though.

    May return NULL.
    """
    return _mupdf.ll_fz_clone_context()