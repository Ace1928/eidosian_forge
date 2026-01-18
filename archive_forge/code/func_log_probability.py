from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def log_probability(self, sequence):
    """
        Returns the log-probability of the given symbol sequence. If the
        sequence is labelled, then returns the joint log-probability of the
        symbol, state sequence. Otherwise, uses the forward algorithm to find
        the log-probability over all label sequences.

        :return: the log-probability of the sequence
        :rtype: float
        :param sequence: the sequence of symbols which must contain the TEXT
            property, and optionally the TAG property
        :type sequence:  Token
        """
    sequence = self._transform(sequence)
    T = len(sequence)
    if T > 0 and sequence[0][_TAG]:
        last_state = sequence[0][_TAG]
        p = self._priors.logprob(last_state) + self._output_logprob(last_state, sequence[0][_TEXT])
        for t in range(1, T):
            state = sequence[t][_TAG]
            p += self._transitions[last_state].logprob(state) + self._output_logprob(state, sequence[t][_TEXT])
            last_state = state
        return p
    else:
        alpha = self._forward_probability(sequence)
        p = logsumexp2(alpha[T - 1])
        return p