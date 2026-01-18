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
def hash_main(alg):
    """Generate hash values for test from standard input."""
    hashmap = {'cy_bytes': ft_hash_bytes}
    try:
        fun = hashmap[alg]
    except KeyError:
        raise KeyError('invalid alg: %r expected one of %r' % (alg, sorted(hashmap)))
    for line in sys.stdin:
        if 'bytes' in alg:
            words = line.encode('utf-8').rstrip().split(b' ')
        else:
            words = line.rstrip().split(' ')
        for word in words:
            print('u%r: %r,' % (word, fun(word)))