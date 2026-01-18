import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def get_semtypes(self, node):
    """
        Based on the node, return a list of plausible semtypes in order of
        plausibility.
        """
    rel = node['rel'].lower()
    word = node['word'].lower()
    if rel == 'spec':
        if word in SPEC_SEMTYPES:
            return [SPEC_SEMTYPES[word]]
        else:
            return [SPEC_SEMTYPES['default']]
    elif rel in ['nmod', 'vmod']:
        return [node['tag'], rel]
    else:
        return [node['tag']]