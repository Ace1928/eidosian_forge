import re
from nltk.corpus.reader import CorpusReader
def senti_synsets(self, string, pos=None):
    from nltk.corpus import wordnet as wn
    sentis = []
    synset_list = wn.synsets(string, pos)
    for synset in synset_list:
        sentis.append(self.senti_synset(synset.name()))
    sentis = filter(lambda x: x, sentis)
    return sentis