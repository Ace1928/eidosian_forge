import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
def find_phrases(self, sentences):
    """Get all unique phrases (multi-word expressions) that appear in ``sentences``, and their scores.

        Parameters
        ----------
        sentences : iterable of list of str
            Text corpus.

        Returns
        -------
        dict(str, float)
           Unique phrases found in ``sentences``, mapped to their scores.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, min_count=1, threshold=0.1, connector_words=ENGLISH_CONNECTOR_WORDS)
            >>>
            >>> for phrase, score in phrases.find_phrases(sentences).items():
            ...     print(phrase, score)
        """
    result = {}
    for sentence in sentences:
        for phrase, score in self.analyze_sentence(sentence):
            if score is not None:
                result[phrase] = score
    return result