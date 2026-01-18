from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
def test_bad_unicode(self):
    buf = io.BytesIO()
    buf.name = 'dummy name to keep fasttext happy'
    buf.write(struct.pack('@3i', 2, -1, -1))
    buf.write(struct.pack('@1q', 10))
    buf.write(b'\xe8\x8b\xb1\xe8\xaa\x9e\xe7\x89\x88\xe3\x82\xa6\xe3\x82\xa3\xe3\x82\xad\xe3\x83\x9a\xe3\x83\x87\xe3\x82\xa3\xe3\x82\xa2\xe3\x81\xb8\xe3\x81\xae\xe6\x8a\x95\xe7\xa8\xbf\xe3\x81\xaf\xe3\x81\x84\xe3\x81\xa4\xe3\x81\xa7\xe3\x82\x82\xe6')
    buf.write(b'\x00')
    buf.write(struct.pack('@qb', 1, -1))
    buf.write(b'\xd0\xb0\xd0\xb4\xd0\xbc\xd0\xb8\xd0\xbd\xd0\xb8\xd1\x81\xd1\x82\xd1\x80\xd0\xb0\xd1\x82\xd0\xb8\xd0\xb2\xd0\xbd\xd0\xbe-\xd1\x82\xd0\xb5\xd1\x80\xd1\x80\xd0\xb8\xd1\x82\xd0\xbe\xd1\x80\xd0\xb8\xd0\xb0\xd0\xbb\xd1\x8c\xd0\xbd\xd1')
    buf.write(b'\x00')
    buf.write(struct.pack('@qb', 2, -1))
    buf.seek(0)
    raw_vocab, vocab_size, nlabels, ntokens = gensim.models._fasttext_bin._load_vocab(buf, False)
    expected = {u'英語版ウィキペディアへの投稿はいつでも\\xe6': 1, u'административно-территориальн\\xd1': 2}
    self.assertEqual(expected, dict(raw_vocab))
    self.assertEqual(vocab_size, 2)
    self.assertEqual(nlabels, -1)
    self.assertEqual(ntokens, 10)