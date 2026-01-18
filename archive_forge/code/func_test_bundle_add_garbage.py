import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_bundle_add_garbage(self):
    document = ProvDocument()

    def test():
        document.add_bundle(document.entity(EX_NS['entity_trying_to_be_a_bundle']))
    self.assertRaises(ProvException, test)

    def test():
        bundle = ProvBundle()
        document.add_bundle(bundle)
    self.assertRaises(ProvException, test)