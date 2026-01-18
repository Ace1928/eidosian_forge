import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
def evaluate_spearman(self, embedding):
    """Evaluate spearman scores for lexical entailment for given embedding.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding for which evaluation is to be done.

        Returns
        -------
        float
            Spearman correlation score for the task for input embedding.

        """
    predicted_scores = []
    expected_scores = []
    skipped = 0
    count = 0
    vocab_trie = self.create_vocab_trie(embedding)
    for (word_1, word_2), expected_score in self.scores.items():
        try:
            predicted_score = self.score_function(embedding, vocab_trie, word_1, word_2)
        except ValueError:
            skipped += 1
            continue
        count += 1
        predicted_scores.append(predicted_score)
        expected_scores.append(expected_score)
    logger.info('skipped pairs: %d out of %d' % (skipped, len(self.scores)))
    spearman = spearmanr(expected_scores, predicted_scores)
    return spearman.correlation