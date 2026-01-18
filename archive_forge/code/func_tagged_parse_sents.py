import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import (
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tree import Tree
def tagged_parse_sents(self, sentences, verbose=False):
    """
        Currently unimplemented because the neural dependency parser (and
        the StanfordCoreNLP pipeline class) doesn't support passing in pre-
        tagged tokens.
        """
    raise NotImplementedError('tagged_parse[_sents] is not supported by StanfordNeuralDependencyParser; use parse[_sents] or raw_parse[_sents] instead.')