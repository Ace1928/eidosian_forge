from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def mark_alls_fresh(self):
    for u, _ in self.sets[Categories.ALL]:
        u._exhausted = False