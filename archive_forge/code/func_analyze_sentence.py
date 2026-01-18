import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
def analyze_sentence(self, sentence):
    """Analyze a sentence, concatenating any detected phrases into a single token.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.

        Yields
        ------
        (str, {float, None})
            Iterate through the input sentence tokens and yield 2-tuples of:
            - ``(concatenated_phrase_tokens, score)`` for token sequences that form a phrase.
            - ``(word, None)`` if the token is not a part of a phrase.

        """
    start_token, in_between = (None, [])
    for word in sentence:
        if word not in self.connector_words:
            if start_token:
                phrase, score = self.score_candidate(start_token, word, in_between)
                if score is not None:
                    yield (phrase, score)
                    start_token, in_between = (None, [])
                else:
                    yield (start_token, None)
                    for w in in_between:
                        yield (w, None)
                    start_token, in_between = (word, [])
            else:
                start_token, in_between = (word, [])
        elif start_token:
            in_between.append(word)
        else:
            yield (word, None)
    if start_token:
        yield (start_token, None)
        for w in in_between:
            yield (w, None)