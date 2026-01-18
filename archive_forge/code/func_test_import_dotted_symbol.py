import unittest
from traits.util.api import import_symbol
def test_import_dotted_symbol(self):
    """ import dotted symbol """
    import tarfile
    symbol = import_symbol('tarfile.TarFile')
    self.assertEqual(symbol, tarfile.TarFile)