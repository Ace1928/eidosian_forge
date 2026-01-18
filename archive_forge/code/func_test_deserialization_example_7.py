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
def test_deserialization_example_7(self):
    """
        Test the deserialization of example 7 which is a simple activity
        description.
        """
    actual_doc = prov.ProvDocument.deserialize(source=os.path.join(DATA_PATH, 'example_07.xml'), format='xml')
    expected_document = prov.ProvDocument()
    ex_ns = Namespace(*EX_NS)
    expected_document.add_namespace(ex_ns)
    expected_document.activity('ex:a1', '2011-11-16T16:05:00', '2011-11-16T16:06:00', [(prov.PROV_TYPE, QualifiedName(ex_ns, 'edit')), ('ex:host', 'server.example.org')])
    self.assertEqual(actual_doc, expected_document)