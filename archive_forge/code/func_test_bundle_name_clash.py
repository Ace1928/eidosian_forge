import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_bundle_name_clash(self):
    document = ProvDocument()

    def test():
        document.bundle(EX_NS['indistinct'])
        document.bundle(EX_NS['indistinct'])
    self.assertRaises(ProvException, test)
    document = ProvDocument()

    def test():
        document.bundle(EX_NS['indistinct'])
        bundle = ProvBundle(identifier=EX_NS['indistinct'])
        document.add_bundle(bundle)
    self.assertRaises(ProvException, test)