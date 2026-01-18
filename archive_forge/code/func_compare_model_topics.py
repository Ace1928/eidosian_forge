from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def compare_model_topics(self, model_topics):
    """Perform the coherence evaluation for each of the models.

        Parameters
        ----------
        model_topics : list of list of str
            list of list of words for the model trained with that number of topics.

        Returns
        -------
        list of (float, float)
            Sequence of pairs of average topic coherence and average coherence.

        Notes
        -----
        This first precomputes the probabilities once, then evaluates coherence for each model.

        Since we have already precomputed the probabilities, this simply involves using the accumulated stats in the
        :class:`~gensim.models.coherencemodel.CoherenceModel` to perform the evaluations, which should be pretty quick.

        """
    orig_topics = self._topics
    orig_topn = self.topn
    try:
        coherences = self._compare_model_topics(model_topics)
    finally:
        self.topics = orig_topics
        self.topn = orig_topn
    return coherences