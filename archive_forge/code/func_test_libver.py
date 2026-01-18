import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
def test_libver(self):
    """ Test libver bounds set/get """
    plist = h5p.create(h5p.FILE_ACCESS)
    plist.set_libver_bounds(h5f.LIBVER_EARLIEST, h5f.LIBVER_LATEST)
    self.assertEqual((h5f.LIBVER_EARLIEST, h5f.LIBVER_LATEST), plist.get_libver_bounds())