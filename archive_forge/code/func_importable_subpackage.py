from functools import partial
from glob import glob
from distutils.util import convert_path
import distutils.command.build_py as orig
import os
import fnmatch
import textwrap
import io
import distutils.errors
import itertools
import stat
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from ..extern.more_itertools import unique_everseen
from ..warnings import SetuptoolsDeprecationWarning
def importable_subpackage(self, parent, file):
    pkg = Path(file).parent
    parts = list(itertools.takewhile(str.isidentifier, pkg.parts))
    if parts:
        return '.'.join([parent, *parts])
    return None