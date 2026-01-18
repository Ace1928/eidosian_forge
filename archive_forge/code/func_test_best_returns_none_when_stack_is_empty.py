import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_best_returns_none_when_stack_is_empty(self):
    stack = _Stack(3)
    self.assertEqual(stack.best(), None)