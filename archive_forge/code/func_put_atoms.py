from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def put_atoms(self, atoms):
    for atom, neg in atoms:
        if neg:
            self[Categories.N_ATOM].add((-atom, None))
        else:
            self[Categories.ATOM].add((atom, None))