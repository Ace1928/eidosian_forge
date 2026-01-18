import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_document_helper_methods(self):
    document = ProvDocument()
    self.assertFalse(document.is_bundle())
    self.assertFalse(document.has_bundles())
    document.bundle(EX_NS['b'])
    self.assertTrue(document.has_bundles())
    self.assertEqual('<ProvDocument>', str(document))