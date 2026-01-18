import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
def test_add_bundle_document(self):
    d1 = self.document_1()
    d2 = self.document_2()

    def sub_test_1():
        d1.add_bundle(d2)
    self.assertRaises(ProvException, sub_test_1)
    ex2_b2 = d2.valid_qualified_name('ex:b2')
    d1.add_bundle(d2, 'ex:b2')
    self.assertEqual(ex2_b2, first(d1.bundles).identifier)
    self.assertNotIn(d2, d1.bundles)
    b2 = ProvBundle()
    b2.update(d2)
    self.assertIn(b2, d1.bundles)