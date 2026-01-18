import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def glob_one(possible_glob):
    """Same as glob.glob().

    work around bugs in glob.glob()
    - Python bug #1001604 ("glob doesn't return unicode with ...")
    - failing expansion for */* with non-iso-8859-* chars
    """
    corrected_glob, corrected = _ensure_with_dir(possible_glob)
    glob_files = glob.glob(corrected_glob)
    if not glob_files:
        glob_files = [possible_glob]
    elif corrected:
        glob_files = [_undo_ensure_with_dir(elem, corrected) for elem in glob_files]
    return [elem.replace('\\', '/') for elem in glob_files]