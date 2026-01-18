from h5py import h5
from .common import TestCase
def test_cnames_get(self):
    cfg = h5.get_config()
    self.assertEqual(cfg.complex_names, ('r', 'i'))