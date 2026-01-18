import difflib
import glob
import inspect
import io
from lxml import etree
import os
import unittest
import warnings
from prov.identifier import Namespace, QualifiedName
from prov.constants import PROV
import prov.model as prov
from prov.tests.test_model import AllTestsBase
from prov.tests.utility import RoundTripTestCase
def test_nested_default_namespace(self):
    """
        Tests that a default namespace that is defined in a lower level tag is
        written to a bundle.
        """
    filename = os.path.join(DATA_PATH, 'nested_default_namespace.xml')
    doc = prov.ProvDocument.deserialize(source=filename, format='xml')
    ns = Namespace('', 'http://example.org/0/')
    self.assertEqual(len(doc._records), 1)
    self.assertEqual(doc.get_default_namespace(), ns)
    self.assertEqual(doc._records[0].identifier.namespace, ns)
    self.assertEqual(doc._records[0].identifier.localpart, 'e001')