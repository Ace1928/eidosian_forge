import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
def export_phrases(self):
    """Extract all found phrases.

        Returns
        ------
        dict(str, float)
            Mapping between phrases and their scores.

        """
    result, source_vocab = ({}, self.vocab)
    for token in source_vocab:
        unigrams = token.split(self.delimiter)
        if len(unigrams) < 2:
            continue
        phrase, score = self.score_candidate(unigrams[0], unigrams[-1], unigrams[1:-1])
        if score is not None:
            result[phrase] = score
    return result