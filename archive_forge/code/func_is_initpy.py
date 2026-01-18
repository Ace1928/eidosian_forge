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
def is_initpy(filename: str) -> bool:
    """Check *filename* is __init__ file or not."""
    basename = path.basename(filename)
    for suffix in sorted(PY_SUFFIXES, key=len, reverse=True):
        if basename == '__init__' + suffix:
            return True
    else:
        return False