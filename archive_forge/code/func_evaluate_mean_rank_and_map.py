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
def evaluate_mean_rank_and_map(self, max_n=None):
    """Evaluate mean rank and MAP for link prediction.

        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.

        Returns
        -------
        tuple (float, float)
            (mean_rank, MAP), e.g (50.3, 0.31).

        """
    ranks = []
    avg_precision_scores = []
    for i, item in enumerate(self.items, start=1):
        if item not in self.relations['unknown']:
            continue
        unknown_relations = list(self.relations['unknown'][item])
        known_relations = list(self.relations['known'][item])
        item_term = self.embedding.index_to_key[item]
        item_distances = self.embedding.distances(item_term)
        unknown_relation_ranks, avg_precision = self.get_unknown_relation_ranks_and_avg_prec(item_distances, unknown_relations, known_relations)
        ranks += unknown_relation_ranks
        avg_precision_scores.append(avg_precision)
        if max_n is not None and i > max_n:
            break
    return (np.mean(ranks), np.mean(avg_precision_scores))