import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
def discourses(self):
    return StreamBackedCorpusView(self.abspath('tagged'), self._discourses_block_reader)