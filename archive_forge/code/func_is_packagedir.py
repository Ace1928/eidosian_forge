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
def is_packagedir(dirname: Optional[str]=None, files: Optional[List[str]]=None) -> bool:
    """Check given *files* contains __init__ file."""
    if files is None and dirname is None:
        return False
    if files is None:
        files = os.listdir(dirname)
    return any((f for f in files if is_initpy(f)))