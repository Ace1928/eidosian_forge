import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_find_all_src_phrases(self):
    phrase_table = TestStackDecoder.create_fake_phrase_table()
    stack_decoder = StackDecoder(phrase_table, None)
    sentence = ('my', 'hovercraft', 'is', 'full', 'of', 'eels')
    src_phrase_spans = stack_decoder.find_all_src_phrases(sentence)
    self.assertEqual(src_phrase_spans[0], [2])
    self.assertEqual(src_phrase_spans[1], [2])
    self.assertEqual(src_phrase_spans[2], [3])
    self.assertEqual(src_phrase_spans[3], [5, 6])
    self.assertFalse(src_phrase_spans[4])
    self.assertEqual(src_phrase_spans[5], [6])