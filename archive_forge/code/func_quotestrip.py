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
def quotestrip(word):
    """Remove quotes and/or double quotes around identifiers."""
    if not word:
        return None
    while word.startswith("'") and word.endswith("'") or (word.startswith('"') and word.endswith('"')):
        word = word[1:-1]
    return word