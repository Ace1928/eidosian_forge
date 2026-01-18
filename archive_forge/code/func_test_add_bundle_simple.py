import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
def test_add_bundle_simple(self):
    d1 = self.document_1()
    b0 = self.bundle_0()

    def sub_test_1():
        d1.add_bundle(b0)
    self.assertRaises(ProvException, sub_test_1)
    self.assertFalse(d1.has_bundles())
    d1.add_bundle(b0, 'ex:b0')
    self.assertTrue(d1.has_bundles())
    self.assertIn(b0, d1.bundles)

    def sub_test_2():
        ex2_b0 = b0.identifier
        d1.add_bundle(ProvBundle(identifier=ex2_b0))
    self.assertRaises(ProvException, sub_test_2)
    d1.add_bundle(ProvBundle(), 'ex:b0')
    self.assertEqual(len(d1.bundles), 2)