import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
def hide_setuptools():
    """
    Remove references to setuptools' modules from sys.modules to allow the
    invocation to import the most appropriate setuptools. This technique is
    necessary to avoid issues such as #315 where setuptools upgrading itself
    would fail to find a function declared in the metadata.
    """
    _distutils_hack = sys.modules.get('_distutils_hack', None)
    if _distutils_hack is not None:
        _distutils_hack._remove_shim()
    modules = filter(_needs_hiding, sys.modules)
    _clear_modules(modules)