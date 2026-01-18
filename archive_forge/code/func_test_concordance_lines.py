import contextlib
import sys
import unittest
from io import StringIO
from nltk.corpus import gutenberg
from nltk.text import Text
def test_concordance_lines(self):
    concordance_out = self.text.concordance_list(self.query, lines=3)
    self.assertEqual(self.list_out[:3], [c.line for c in concordance_out])