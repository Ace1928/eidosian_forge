import re
from nltk.corpus.reader import CorpusReader
def all_senti_synsets(self):
    from nltk.corpus import wordnet as wn
    for key, fields in self._db.items():
        pos, offset = key
        pos_score, neg_score = fields
        synset = wn.synset_from_pos_and_offset(pos, offset)
        yield SentiSynset(pos_score, neg_score, synset)