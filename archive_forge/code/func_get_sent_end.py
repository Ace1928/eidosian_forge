import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
def get_sent_end(self, end_word):
    splitted = end_word.split(')')[0].split(',')
    return int(splitted[1]) + int(splitted[2])