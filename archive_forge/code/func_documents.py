import os
import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import RegexpTokenizer
def documents(self, fileids=None):
    """
        Return a list of document identifiers for all documents in
        this corpus, or for the documents with the given file(s) if
        specified.
        """
    if fileids is None:
        return self._documents
    if isinstance(fileids, str):
        fileids = [fileids]
    for f in fileids:
        if f not in self._fileids:
            raise KeyError('File id %s not found' % fileids)
    return sorted({f[:-4] for f in fileids})