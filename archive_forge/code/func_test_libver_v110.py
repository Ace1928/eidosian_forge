import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
def test_libver_v110(self):
    """ Test libver bounds set/get for H5F_LIBVER_V110"""
    plist = h5p.create(h5p.FILE_ACCESS)
    plist.set_libver_bounds(h5f.LIBVER_V18, h5f.LIBVER_V110)
    self.assertEqual((h5f.LIBVER_V18, h5f.LIBVER_V110), plist.get_libver_bounds())