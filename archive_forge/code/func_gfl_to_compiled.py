import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def gfl_to_compiled(self, gfl):
    index_counter = Counter()
    return_list = []
    for gf in gfl:
        return_list.extend(gf.compile(index_counter))
    if self.verbose:
        print('Compiled Glue Premises:')
        for cgf in return_list:
            print(cgf)
    return return_list