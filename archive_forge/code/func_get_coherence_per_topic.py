from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def get_coherence_per_topic(self, segmented_topics=None, with_std=False, with_support=False):
    """Get list of coherence values for each topic based on pipeline parameters.

        Parameters
        ----------
        segmented_topics : list of list of (int, number)
            Topics.
        with_std : bool, optional
            True to also include standard deviation across topic segment sets in addition to the mean coherence
            for each topic.
        with_support : bool, optional
            True to also include support across topic segments. The support is defined as the number of pairwise
            similarity comparisons were used to compute the overall topic coherence.

        Return
        ------
        list of float
            Sequence of similarity measure for each topic.

        """
    measure = self.measure
    if segmented_topics is None:
        segmented_topics = measure.seg(self.topics)
    if self._accumulator is None:
        self.estimate_probabilities(segmented_topics)
    kwargs = dict(with_std=with_std, with_support=with_support)
    if self.coherence in BOOLEAN_DOCUMENT_BASED or self.coherence == 'c_w2v':
        pass
    elif self.coherence == 'c_v':
        kwargs['topics'] = self.topics
        kwargs['measure'] = 'nlr'
        kwargs['gamma'] = 1
    else:
        kwargs['normalize'] = self.coherence == 'c_npmi'
    return measure.conf(segmented_topics, self._accumulator, **kwargs)