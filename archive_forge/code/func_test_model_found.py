import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
def test_model_found(arguments):
    """
    Try some proofs and exhibit the results.
    """
    for goal, assumptions in arguments:
        g = Expression.fromstring(goal)
        alist = [lp.parse(a) for a in assumptions]
        m = MaceCommand(g, assumptions=alist, max_models=50)
        found = m.build_model()
        for a in alist:
            print('   %s' % a)
        print(f'|- {g}: {decode_result(found)}\n')