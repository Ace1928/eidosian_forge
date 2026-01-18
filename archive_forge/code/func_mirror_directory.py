import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
def mirror_directory(source, destination):
    """
    Copy contents of the source directory into destination and fix permissions.

    Parameters
    ----------
    source : str, :class:`pathlib.Path`
        Source data directory.
    destination : str, :class:`pathlib.Path`
        Destination directory that will contain the copy of source. The actual
        source directory (not just it's contents) is copied.

    Returns
    -------
    mirror : :class:`pathlib.Path`
        The path of the mirrored output directory.

    """
    source = Path(source)
    mirror = Path(destination) / source.name
    shutil.copytree(source, mirror)
    _recursive_chmod_directories(mirror, mode=stat.S_IWUSR)
    return mirror