import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
def init_accumulator2(self):
    return self.accumulator_cls(self.top_ids2, self.dictionary2)