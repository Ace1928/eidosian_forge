from abc import ABCMeta, abstractmethod
from nltk.probability import DictionaryProbDist
def cluster_names(self):
    """
        Returns the names of the clusters.
        :rtype: list
        """
    return list(range(self.num_clusters()))