import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
def test_literal_provn_with_single_quotes(self):
    l = Literal('{"foo": "bar"}')
    string_rep = l.provn_representation()
    self.assertTrue('{\\"f' in string_rep)