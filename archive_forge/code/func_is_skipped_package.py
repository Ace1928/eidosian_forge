import argparse
import glob
import locale
import os
import sys
from copy import copy
from fnmatch import fnmatch
from importlib.machinery import EXTENSION_SUFFIXES
from os import path
from typing import Any, Generator, List, Optional, Tuple
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.cmd.quickstart import EXTENSIONS
from sphinx.locale import __
from sphinx.util.osutil import FileAvoidWrite, ensuredir
from sphinx.util.template import ReSTRenderer
def is_skipped_package(dirname: str, opts: Any, excludes: List[str]=[]) -> bool:
    """Check if we want to skip this module."""
    if not path.isdir(dirname):
        return False
    files = glob.glob(path.join(dirname, '*.py'))
    regular_package = any((f for f in files if is_initpy(f)))
    if not regular_package and (not opts.implicit_namespaces):
        return True
    if all((is_excluded(path.join(dirname, f), excludes) for f in files)):
        return True
    else:
        return False