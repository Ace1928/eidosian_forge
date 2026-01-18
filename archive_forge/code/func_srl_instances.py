import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def srl_instances(self, fileids=None, pos_in_tree=None, flatten=True):
    self._require(self.WORDS, self.POS, self.TREE, self.SRL)
    if pos_in_tree is None:
        pos_in_tree = self._pos_in_tree

    def get_srl_instances(grid):
        return self._get_srl_instances(grid, pos_in_tree)
    result = LazyMap(get_srl_instances, self._grids(fileids))
    if flatten:
        result = LazyConcatenation(result)
    return result