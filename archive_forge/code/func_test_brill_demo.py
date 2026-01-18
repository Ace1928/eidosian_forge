import unittest
from nltk.corpus import treebank
from nltk.tag import UnigramTagger, brill, brill_trainer
from nltk.tbl import demo
@unittest.skip('Should be tested in __main__ of nltk.tbl.demo')
def test_brill_demo(self):
    demo()