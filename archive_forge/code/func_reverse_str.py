import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
def reverse_str(str):
    lst = list(str)
    lst.reverse()
    return ''.join(lst)