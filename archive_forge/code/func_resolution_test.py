import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def resolution_test(e):
    f = Expression.fromstring(e)
    t = ResolutionProver().prove(f)
    print(f'|- {f}: {t}')