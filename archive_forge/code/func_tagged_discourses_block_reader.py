import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
def tagged_discourses_block_reader(stream):
    return self._tagged_discourses_block_reader(stream, tagset)