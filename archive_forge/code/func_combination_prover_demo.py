from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def combination_prover_demo():
    lexpr = Expression.fromstring
    p1 = lexpr('see(Socrates, John)')
    p2 = lexpr('see(John, Mary)')
    c = lexpr('-see(Socrates, Mary)')
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    command = ClosedDomainProver(UniqueNamesProver(ClosedWorldProver(prover)))
    for a in command.assumptions():
        print(a)
    print(command.prove())