import errno
import os
import shutil
from . import controldir, errors, ui
from .i18n import gettext
from .osutils import isdir
from .trace import note
from .workingtree import WorkingTree
def is_detritus(subp):
    """Return True if the supplied path is detritus, False otherwise"""
    return subp.endswith('.THIS') or subp.endswith('.BASE') or subp.endswith('.OTHER') or subp.endswith('~') or subp.endswith('.tmp')