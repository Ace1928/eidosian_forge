import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def depgraph_to_glue(self, depgraph):
    return self.get_glue_dict().to_glueformula_list(depgraph)