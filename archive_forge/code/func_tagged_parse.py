import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import (
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tree import Tree
def tagged_parse(self, sentence, verbose=False):
    """
        Use StanfordParser to parse a sentence. Takes a sentence as a list of
        (word, tag) tuples; the sentence must have already been tokenized and
        tagged.

        :param sentence: Input sentence to parse
        :type sentence: list(tuple(str, str))
        :rtype: iter(Tree)
        """
    return next(self.tagged_parse_sents([sentence], verbose))