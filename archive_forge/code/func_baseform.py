import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
@property
def baseform(self):
    """The baseform of the predicate."""
    return self.roleset.split('.')[0]