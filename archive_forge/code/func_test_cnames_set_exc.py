from h5py import h5
from .common import TestCase
def test_cnames_set_exc(self):
    self.addCleanup(fixnames)
    cfg = h5.get_config()
    with self.assertRaises(TypeError):
        cfg.complex_names = ('q', 'i', 'v')
    self.assertEqual(cfg.complex_names, ('r', 'i'))