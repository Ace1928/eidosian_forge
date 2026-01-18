import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from gensim import interfaces, utils, matutils
from gensim.matutils import (
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback
def get_term_topics(self, word_id, minimum_probability=None):
    """Get the most relevant topics to the given word.

        Parameters
        ----------
        word_id : int
            The word for which the topic distribution will be computed.
        minimum_probability : float, optional
            Topics with an assigned probability below this threshold will be discarded.

        Returns
        -------
        list of (int, float)
            The relevant topics represented as pairs of their ID and their assigned probability, sorted
            by relevance to the given word.

        """
    if minimum_probability is None:
        minimum_probability = self.minimum_probability
    minimum_probability = max(minimum_probability, 1e-08)
    if isinstance(word_id, str):
        word_id = self.id2word.doc2bow([word_id])[0][0]
    values = []
    for topic_id in range(0, self.num_topics):
        if self.expElogbeta[topic_id][word_id] >= minimum_probability:
            values.append((topic_id, self.expElogbeta[topic_id][word_id]))
    return values