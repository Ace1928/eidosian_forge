from abc import ABCMeta, abstractmethod
from nltk.probability import DictionaryProbDist
def classification_probdist(self, vector):
    """
        Classifies the token into a cluster, returning
        a probability distribution over the cluster identifiers.
        """
    likelihoods = {}
    sum = 0.0
    for cluster in self.cluster_names():
        likelihoods[cluster] = self.likelihood(vector, cluster)
        sum += likelihoods[cluster]
    for cluster in self.cluster_names():
        likelihoods[cluster] /= sum
    return DictionaryProbDist(likelihoods)