from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
def corpus_from_lines(self, lines):
    fpath = tempfile.mktemp()
    with codecs.open(fpath, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))
    return self.corpus_class(fpath)