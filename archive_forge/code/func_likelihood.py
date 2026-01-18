from abc import ABCMeta, abstractmethod
from nltk.probability import DictionaryProbDist
def likelihood(self, vector, label):
    """
        Returns the likelihood (a float) of the token having the
        corresponding cluster.
        """
    if self.classify(vector) == label:
        return 1.0
    else:
        return 0.0