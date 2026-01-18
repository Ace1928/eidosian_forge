import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
def compute_context_vector(self, segment_word_ids, topic_word_ids):
    """Check if (segment_word_ids, topic_word_ids) context vector has been cached.

        Parameters
        ----------
        segment_word_ids: list
            Ids of words in segment.
        topic_word_ids: list
            Ids of words in topic.
        Returns
        -------
        csr_matrix :class:`~scipy.sparse.csr`
            If context vector has been cached, then return corresponding context vector,
            else compute, cache, and return.

        """
    key = _key_for_segment(segment_word_ids, topic_word_ids)
    context_vector = self.context_vector_cache.get(key, None)
    if context_vector is None:
        context_vector = self._make_seg(segment_word_ids, topic_word_ids)
        self.context_vector_cache[key] = context_vector
    return context_vector