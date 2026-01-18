import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def iob_words(self, fileids=None, tagset=None):
    """
        :return: a list of word/tag/IOB tuples
        :rtype: list(tuple)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
    self._require(self.WORDS, self.POS, self.CHUNK)

    def get_iob_words(grid):
        return self._get_iob_words(grid, tagset)
    return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))