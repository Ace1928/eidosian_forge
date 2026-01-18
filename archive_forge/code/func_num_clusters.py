from abc import ABCMeta, abstractmethod
from nltk.probability import DictionaryProbDist
@abstractmethod
def num_clusters(self):
    """
        Returns the number of clusters.
        """