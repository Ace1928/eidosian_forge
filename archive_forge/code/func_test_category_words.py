import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
def test_category_words(self):
    self.assertEqual(ptb.words(categories=['humor', 'fiction'])[:6], ['Thirty-three', 'Scotty', 'did', 'not', 'go', 'back'])