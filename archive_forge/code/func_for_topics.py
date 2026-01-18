from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
@classmethod
def for_topics(cls, topics_as_topn_terms, **kwargs):
    """Initialize a CoherenceModel with estimated probabilities for all of the given topics.

        Parameters
        ----------
        topics_as_topn_terms : list of list of str
            Each element in the top-level list should be the list of topics for a model.
            The topics for the model should be a list of top-N words, one per topic.

        Return
        ------
        :class:`~gensim.models.coherencemodel.CoherenceModel`
            CoherenceModel with estimated probabilities for all of the given models.

        """
    if not topics_as_topn_terms:
        raise ValueError('len(topics) must be > 0.')
    if any((len(topic_lists) == 0 for topic_lists in topics_as_topn_terms)):
        raise ValueError('found empty topic listing in `topics`')
    topn = 0
    for topic_list in topics_as_topn_terms:
        for topic in topic_list:
            topn = max(topn, len(topic))
    topn = min(kwargs.pop('topn', topn), topn)
    super_topic = utils.flatten(topics_as_topn_terms)
    logging.info('Number of relevant terms for all %d models: %d', len(topics_as_topn_terms), len(super_topic))
    cm = CoherenceModel(topics=[super_topic], topn=len(super_topic), **kwargs)
    cm.estimate_probabilities()
    cm.topn = topn
    return cm