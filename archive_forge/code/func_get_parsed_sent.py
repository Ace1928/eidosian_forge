import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def get_parsed_sent(grid):
    return self._get_parsed_sent(grid, pos_in_tree, tagset)