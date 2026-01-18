from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def print_proof(goal, premises):
    lexpr = Expression.fromstring
    prover = Prover9Command(lexpr(goal), premises)
    command = UniqueNamesProver(ClosedWorldProver(prover))
    print(goal, prover.prove(), command.prove())