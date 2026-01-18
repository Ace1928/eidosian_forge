from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def get_coherence(self):
    """Get coherence value based on pipeline parameters.

        Returns
        -------
        float
            Value of coherence.

        """
    confirmed_measures = self.get_coherence_per_topic()
    return self.aggregate_measures(confirmed_measures)