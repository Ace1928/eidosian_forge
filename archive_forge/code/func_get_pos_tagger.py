import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def get_pos_tagger(self):
    from nltk.corpus import brown
    regexp_tagger = RegexpTagger([('^-?[0-9]+(\\.[0-9]+)?$', 'CD'), ('(The|the|A|a|An|an)$', 'AT'), ('.*able$', 'JJ'), ('.*ness$', 'NN'), ('.*ly$', 'RB'), ('.*s$', 'NNS'), ('.*ing$', 'VBG'), ('.*ed$', 'VBD'), ('.*', 'NN')])
    brown_train = brown.tagged_sents(categories='news')
    unigram_tagger = UnigramTagger(brown_train, backoff=regexp_tagger)
    bigram_tagger = BigramTagger(brown_train, backoff=unigram_tagger)
    trigram_tagger = TrigramTagger(brown_train, backoff=bigram_tagger)
    main_tagger = RegexpTagger([('(A|a|An|an)$', 'ex_quant'), ('(Every|every|All|all)$', 'univ_quant')], backoff=trigram_tagger)
    return main_tagger