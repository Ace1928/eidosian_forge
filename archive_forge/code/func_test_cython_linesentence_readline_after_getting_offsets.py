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
def test_cython_linesentence_readline_after_getting_offsets(self):
    lines = ['line1\n', 'line2\n', 'line3\n', 'line4\n', 'line5\n']
    tmpf = get_tmpfile('gensim_doc2vec.tst')
    with utils.open(tmpf, 'wb', encoding='utf8') as fout:
        for line in lines:
            fout.write(utils.any2unicode(line))
    from gensim.models.word2vec_corpusfile import CythonLineSentence
    offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 5)
    for offset, line in zip(offsets, lines):
        ls = CythonLineSentence(tmpf, offset)
        sentence = ls.read_sentence()
        self.assertEqual(len(sentence), 1)
        self.assertEqual(sentence[0], utils.any2utf8(line.strip()))