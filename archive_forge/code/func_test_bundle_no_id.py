import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_bundle_no_id(self):
    document = ProvDocument()

    def test():
        bundle = ProvBundle()
        document.add_bundle(bundle)
    self.assertRaises(ProvException, test)