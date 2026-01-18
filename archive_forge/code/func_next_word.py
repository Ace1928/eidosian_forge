from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def next_word(self):
    """Return the next NEXUS word from a string.

        This deals with single and double quotes, whitespace and punctuation.
        """
    word = []
    quoted = False
    first = self.next_nonwhitespace()
    if not first:
        return None
    word.append(first)
    if first == "'":
        quoted = "'"
    elif first == '"':
        quoted = '"'
    elif first in PUNCTUATION:
        return first
    while True:
        c = self.peek()
        if c == quoted:
            word.append(next(self))
            if self.peek() == quoted:
                next(self)
            elif quoted:
                break
        elif quoted:
            word.append(next(self))
        elif not c or c in PUNCTUATION or c in WHITESPACE:
            break
        else:
            word.append(next(self))
    return ''.join(word)