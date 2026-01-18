import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def target_update(target, deps, cmd):
    """Update a target with a given command given a list of dependencies.

    target_update(target,deps,cmd) -> runs cmd if target is outdated.

    This is just a wrapper around target_outdated() which calls the given
    command if target is outdated.

    .. deprecated:: 8.22
    """
    warnings.warn('`target_update` is deprecated since IPython 8.22 and will be removed in future versions', DeprecationWarning, stacklevel=2)
    if target_outdated(target, deps):
        system(cmd)