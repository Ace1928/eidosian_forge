import sys
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
def parsed_words(self, fileids=None, categories=None):
    return super().parsed_words(self._resolve(fileids, categories))