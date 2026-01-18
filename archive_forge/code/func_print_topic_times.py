import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def print_topic_times(self, topic, top_terms=20):
    """Get the most relevant words for a topic, for each timeslice. This can be used to inspect the evolution of a
        topic through time.

        Parameters
        ----------
        topic : int
            The index of the topic.
        top_terms : int, optional
            Number of most relevant words associated with the topic to be returned.

        Returns
        -------
        list of list of str
            Top `top_terms` relevant terms for the topic for each time slice.

        """
    topics = []
    for time in range(self.num_time_slices):
        topics.append(self.print_topic(topic, time, top_terms))
    return topics