import unittest
from nltk.corpus import nombank
def test_framefiles_fileids(self):
    self.assertEqual(len(nombank.fileids()), 4705)
    self.assertTrue(all((fileid.endswith('.xml') for fileid in nombank.fileids())))