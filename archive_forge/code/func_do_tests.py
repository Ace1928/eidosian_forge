import io
import logging
import unittest
from prov.model import ProvDocument
def do_tests(self, prov_doc, msg=None):
    self.assertRoundTripEquivalence(prov_doc, msg)