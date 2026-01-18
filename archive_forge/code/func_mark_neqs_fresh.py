from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def mark_neqs_fresh(self):
    for neq, _ in self.sets[Categories.N_EQ]:
        neq._exhausted = False