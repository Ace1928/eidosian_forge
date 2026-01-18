import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
def data_section(self, tokens, labeled=None):
    """
        Returns the ARFF data section for the given data.

        :param tokens: a list of featuresets (dicts) or labelled featuresets
            which are tuples (featureset, label).
        :param labeled: Indicates whether the given tokens are labeled
            or not.  If None, then the tokens will be assumed to be
            labeled if the first token's value is a tuple or list.
        """
    if labeled is None:
        labeled = tokens and isinstance(tokens[0], (tuple, list))
    if not labeled:
        tokens = [(tok, None) for tok in tokens]
    s = '\n@DATA\n'
    for tok, label in tokens:
        for fname, ftype in self._features:
            s += '%s,' % self._fmt_arff_val(tok.get(fname))
        s += '%s\n' % self._fmt_arff_val(label)
    return s