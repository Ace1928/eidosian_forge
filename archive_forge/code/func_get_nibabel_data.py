import unittest
from os import environ, listdir
from os.path import dirname, exists, isdir
from os.path import join as pjoin
from os.path import realpath
def get_nibabel_data():
    """Return path to nibabel-data or empty string if missing

    First use ``NIBABEL_DATA_DIR`` environment variable.

    If this variable is missing then look for data in directory below package
    directory.
    """
    nibabel_data = environ.get('NIBABEL_DATA_DIR')
    if nibabel_data is None:
        mod = __import__('nibabel')
        containing_path = dirname(dirname(realpath(mod.__file__)))
        nibabel_data = pjoin(containing_path, 'nibabel-data')
    return nibabel_data if isdir(nibabel_data) else ''