from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def closed_world_demo():
    lexpr = Expression.fromstring
    p1 = lexpr('walk(Socrates)')
    p2 = lexpr('(Socrates != Bill)')
    c = lexpr('-walk(Bill)')
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print('assumptions:')
    for a in cwp.assumptions():
        print('   ', a)
    print('goal:', cwp.goal())
    print(cwp.prove())
    p1 = lexpr('see(Socrates, John)')
    p2 = lexpr('see(John, Mary)')
    p3 = lexpr('(Socrates != John)')
    p4 = lexpr('(John != Mary)')
    c = lexpr('-see(Socrates, Mary)')
    prover = Prover9Command(c, [p1, p2, p3, p4])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print('assumptions:')
    for a in cwp.assumptions():
        print('   ', a)
    print('goal:', cwp.goal())
    print(cwp.prove())
    p1 = lexpr('all x.(ostrich(x) -> bird(x))')
    p2 = lexpr('bird(Tweety)')
    p3 = lexpr('-ostrich(Sam)')
    p4 = lexpr('Sam != Tweety')
    c = lexpr('-bird(Sam)')
    prover = Prover9Command(c, [p1, p2, p3, p4])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print('assumptions:')
    for a in cwp.assumptions():
        print('   ', a)
    print('goal:', cwp.goal())
    print(cwp.prove())