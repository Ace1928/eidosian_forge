from h5py import h5
from .common import TestCase
def test_cnames_set(self):
    self.addCleanup(fixnames)
    cfg = h5.get_config()
    cfg.complex_names = ('q', 'x')
    self.assertEqual(cfg.complex_names, ('q', 'x'))