import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
def init_accumulator(self):
    return self.accumulator_cls(self.top_ids, self.dictionary)