from nltk.internals import overridden
def prob_classify_many(self, featuresets):
    """
        Apply ``self.prob_classify()`` to each element of ``featuresets``.  I.e.:

            return [self.prob_classify(fs) for fs in featuresets]

        :rtype: list(ProbDistI)
        """
    return [self.prob_classify(fs) for fs in featuresets]