import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
def test_entries(self):
    self.assertEqual(mwa_ppdb.entries()[:10], [('10/17/01', '17/10/2001'), ('102,70', '102.70'), ('13,53', '13.53'), ('3.2.5.3.2.1', '3.2.5.3.2.1.'), ('53,76', '53.76'), ('6.9.5', '6.9.5.'), ('7.7.6.3', '7.7.6.3.'), ('76,20', '76.20'), ('79,85', '79.85'), ('93,65', '93.65')])