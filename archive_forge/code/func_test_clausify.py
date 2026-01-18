import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def test_clausify():
    lexpr = Expression.fromstring
    print(clausify(lexpr('P(x) | Q(x)')))
    print(clausify(lexpr('(P(x) & Q(x)) | R(x)')))
    print(clausify(lexpr('P(x) | (Q(x) & R(x))')))
    print(clausify(lexpr('(P(x) & Q(x)) | (R(x) & S(x))')))
    print(clausify(lexpr('P(x) | Q(x) | R(x)')))
    print(clausify(lexpr('P(x) | (Q(x) & R(x)) | S(x)')))
    print(clausify(lexpr('exists x.P(x) | Q(x)')))
    print(clausify(lexpr('-(-P(x) & Q(x))')))
    print(clausify(lexpr('P(x) <-> Q(x)')))
    print(clausify(lexpr('-(P(x) <-> Q(x))')))
    print(clausify(lexpr('-(all x.P(x))')))
    print(clausify(lexpr('-(some x.P(x))')))
    print(clausify(lexpr('some x.P(x)')))
    print(clausify(lexpr('some x.all y.P(x,y)')))
    print(clausify(lexpr('all y.some x.P(x,y)')))
    print(clausify(lexpr('all z.all y.some x.P(x,y,z)')))
    print(clausify(lexpr('all x.(all y.P(x,y) -> -all y.(Q(x,y) -> R(x,y)))')))