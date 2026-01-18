from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def test_mixed_tag_types(self):
    """Ensure alternating int/string tags don't share indexes in vectors"""
    mixed_tag_corpus = [doc2vec.TaggedDocument(words, [i, words[0]]) for i, words in enumerate(raw_sentences)]
    model = doc2vec.Doc2Vec()
    model.build_vocab(mixed_tag_corpus)
    expected_length = len(sentences) + len(model.dv.key_to_index)
    self.assertEqual(len(model.dv.vectors), expected_length)