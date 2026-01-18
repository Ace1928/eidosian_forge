import unittest
import nltk.data
from nltk.corpus.reader.util import (
def test_correct_length(self):
    for f, file_data in self.data():
        v = StreamBackedCorpusView(f, read_whitespace_block)
        self.assertEqual(len(v), len(file_data.split()))
        v = StreamBackedCorpusView(f, read_line_block)
        self.assertEqual(len(v), len(self.linetok.tokenize(file_data)))