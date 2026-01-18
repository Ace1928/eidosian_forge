import re
from nltk.corpus.reader import CorpusReader
def senti_synset(self, *vals):
    from nltk.corpus import wordnet as wn
    if tuple(vals) in self._db:
        pos_score, neg_score = self._db[tuple(vals)]
        pos, offset = vals
        if pos == 's':
            pos = 'a'
        synset = wn.synset_from_pos_and_offset(pos, offset)
        return SentiSynset(pos_score, neg_score, synset)
    else:
        synset = wn.synset(vals[0])
        pos = synset.pos()
        if pos == 's':
            pos = 'a'
        offset = synset.offset()
        if (pos, offset) in self._db:
            pos_score, neg_score = self._db[pos, offset]
            return SentiSynset(pos_score, neg_score, synset)
        else:
            return None