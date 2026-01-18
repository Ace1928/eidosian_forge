from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def testHigherOrderTableauProver():
    tableau_test('believe(j, -lie(b))', ['believe(j, -lie(b) & -cheat(b))'])
    tableau_test('believe(j, lie(b) & cheat(b))', ['believe(j, lie(b))'])
    tableau_test('believe(j, lie(b))', ['lie(b)'])
    tableau_test('believe(j, know(b, cheat(b)))', ['believe(j, know(b, lie(b)) & know(b, steals(b) & cheat(b)))'])
    tableau_test('P(Q(y), R(y) & R(z))', ['P(Q(x) & Q(y), R(y) & R(z))'])
    tableau_test('believe(j, cheat(b) & lie(b))', ['believe(j, lie(b) & cheat(b))'])
    tableau_test('believe(j, -cheat(b) & -lie(b))', ['believe(j, -lie(b) & -cheat(b))'])