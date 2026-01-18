import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
def test_make_relation_set():
    print(MaceCommand._make_relation_set(num_entities=3, values=[1, 0, 1]) == {('c',), ('a',)})
    print(MaceCommand._make_relation_set(num_entities=3, values=[0, 0, 0, 0, 0, 0, 1, 0, 0]) == {('c', 'a')})
    print(MaceCommand._make_relation_set(num_entities=2, values=[0, 0, 1, 0, 0, 0, 1, 0]) == {('a', 'b', 'a'), ('b', 'b', 'a')})