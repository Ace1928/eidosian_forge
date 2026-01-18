import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def recluster(self, eps=0.1, min_samples=None, min_cores=None):
    """Reapply CBDBSCAN clustering and stable topic generation.

        Stable topics can be retrieved using :meth:`~gensim.models.ensemblelda.EnsembleLda.get_topics`.

        Parameters
        ----------
        eps : float
            epsilon for the CBDBSCAN algorithm, having the same meaning as in classic DBSCAN clustering.
            default: ``0.1``
        min_samples : int
            The minimum number of samples in the neighborhood of a topic to be considered a core in CBDBSCAN.
            default: ``int(self.num_models / 2)``
        min_cores : int
            how many cores a cluster has to have, to be treated as stable topic. That means, how many topics
            that look similar have to be present, so that the average topic in those is used as stable topic.
            default: ``min(3, max(1, int(self.num_models /4 +1)))``

        """
    if self.asymmetric_distance_matrix_outdated:
        logger.info('asymmetric distance matrix is outdated due to add_model')
        self._generate_asymmetric_distance_matrix()
    self._generate_topic_clusters(eps, min_samples)
    self._generate_stable_topics(min_cores)
    self.generate_gensim_representation()