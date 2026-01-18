import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
def document_2(self):
    d2 = ProvDocument()
    ns_ex = d2.add_namespace('ex', EX2_URI)
    d2.activity(ns_ex['a1'])
    return d2