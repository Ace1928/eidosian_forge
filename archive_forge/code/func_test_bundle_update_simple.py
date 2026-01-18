import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
def test_bundle_update_simple(self):
    doc = ProvDocument()
    doc.set_default_namespace(EX_URI)
    b1 = doc.bundle('b1')
    b1.entity('e')
    b2 = doc.bundle('b2')
    b2.entity('e')
    self.assertRaises(ProvException, lambda: b1.update(1))
    self.assertRaises(ProvException, lambda: b1.update(doc))
    b1.update(b2)
    self.assertEqual(len(b1.get_records()), 2)