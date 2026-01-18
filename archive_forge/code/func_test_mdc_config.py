import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
def test_mdc_config(self):
    """test get/set mdc config """
    falist = h5p.create(h5p.FILE_ACCESS)
    config = falist.get_mdc_config()
    falist.set_mdc_config(config)