import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
def get_sent_beg(self, beg_word):
    return int(beg_word.split(',')[1])