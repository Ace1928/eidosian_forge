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
def ngram_main():
    """Generate ngrams for tests from standard input."""
    alg = sys.argv[1]
    minn = int(sys.argv[2])
    maxn = int(sys.argv[3])
    assert minn <= maxn, 'expected sane command-line parameters'
    hashmap = {'cy_text': compute_ngrams, 'cy_bytes': compute_ngrams_bytes}
    try:
        fun = hashmap[alg]
    except KeyError:
        raise KeyError('invalid alg: %r expected one of %r' % (alg, sorted(hashmap)))
    for line in sys.stdin:
        word = line.rstrip('\n')
        ngrams = fun(word, minn, maxn)
        print('%r: %r,' % (word, ngrams))