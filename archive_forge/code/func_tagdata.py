from math import log
from operator import itemgetter
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI
def tagdata(self, data):
    """
        Tags each sentence in a list of sentences

        :param data:list of list of words
        :type data: [[string,],]
        :return: list of list of (word, tag) tuples

        Invokes tag(sent) function for each sentence
        compiles the results into a list of tagged sentences
        each tagged sentence is a list of (word, tag) tuples
        """
    res = []
    for sent in data:
        res1 = self.tag(sent)
        res.append(res1)
    return res