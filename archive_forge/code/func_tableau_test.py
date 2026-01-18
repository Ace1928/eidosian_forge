from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def tableau_test(c, ps=None, verbose=False):
    pc = Expression.fromstring(c)
    pps = [Expression.fromstring(p) for p in ps] if ps else []
    if not ps:
        ps = []
    print('%s |- %s: %s' % (', '.join(ps), pc, TableauProver().prove(pc, pps, verbose=verbose)))