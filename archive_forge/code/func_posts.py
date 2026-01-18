import re
import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
from nltk.internals import ElementWrapper
from nltk.tag import map_tag
from nltk.util import LazyConcatenation
def posts(self, fileids=None):
    return concat([XMLCorpusView(fileid, 'Session/Posts/Post/terminals', self._elt_to_words) for fileid in self.abspaths(fileids)])