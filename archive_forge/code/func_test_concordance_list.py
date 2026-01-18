import contextlib
import sys
import unittest
from io import StringIO
from nltk.corpus import gutenberg
from nltk.text import Text
def test_concordance_list(self):
    concordance_out = self.text.concordance_list(self.query)
    self.assertEqual(self.list_out, [c.line for c in concordance_out])