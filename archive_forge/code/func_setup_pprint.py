import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def setup_pprint():
    from sympy.interactive.printing import init_printing
    from sympy.printing.pretty.pretty import pprint_use_unicode
    import sympy.interactive.printing as interactive_printing
    use_unicode_prev = pprint_use_unicode(False)
    init_printing(pretty_print=False)
    interactive_printing.NO_GLOBAL = True
    return use_unicode_prev