import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_bundle_get_records(self):
    document = ProvDocument()
    document.entity(identifier=EX_NS['e1'])
    document.agent(identifier=EX_NS['e1'])
    self.assertEqual(len(list(document.get_records(ProvAgent))), 1)
    self.assertEqual(len(document.get_records()), 2)