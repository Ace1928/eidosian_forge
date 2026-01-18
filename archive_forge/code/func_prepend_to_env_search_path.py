from __future__ import annotations
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from traitlets import Bool, Instance, Integer, List, Unicode, default
from nbconvert.utils import _contextlib_chdir
from .latex import LatexExporter
def prepend_to_env_search_path(varname, value, envdict):
    """Add value to the environment variable varname in envdict

    e.g. prepend_to_env_search_path('BIBINPUTS', '/home/sally/foo', os.environ)
    """
    if not value:
        return
    envdict[varname] = value + os.pathsep + envdict.get(varname, '')