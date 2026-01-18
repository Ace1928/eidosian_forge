import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
def test_news_fileids(self):
    self.assertEqual(ptb.fileids('news')[:3], ['WSJ/00/WSJ_0001.MRG', 'WSJ/00/WSJ_0002.MRG', 'WSJ/00/WSJ_0003.MRG'])