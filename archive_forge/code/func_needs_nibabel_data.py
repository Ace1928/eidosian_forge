import unittest
from os import environ, listdir
from os.path import dirname, exists, isdir
from os.path import join as pjoin
from os.path import realpath
def needs_nibabel_data(subdir=None):
    """Decorator for tests needing nibabel-data

    Parameters
    ----------
    subdir : None or str
        Subdirectory we need in nibabel-data directory.  If None, only require
        nibabel-data directory itself.

    Returns
    -------
    skip_dec : decorator
        Decorator skipping tests if required directory not present
    """
    nibabel_data = get_nibabel_data()
    if nibabel_data == '':
        return unittest.skip('Need nibabel-data directory for this test')
    if subdir is None:
        return lambda x: x
    required_path = pjoin(nibabel_data, subdir)
    have_files = exists(required_path) and len(listdir(required_path)) > 0
    return unittest.skipUnless(have_files, f'Need files in {required_path} for these tests')