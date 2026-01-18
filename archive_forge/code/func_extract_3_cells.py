import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def extract_3_cells(cells, index):
    word, tag, head = cells
    return (index, word, word, tag, tag, '', head, '')