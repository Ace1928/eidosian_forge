import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def find_answers(self, verbose=False):
    self.prove(verbose)
    answers = set()
    answer_ex = VariableExpression(Variable(ResolutionProver.ANSWER_KEY))
    for clause in self._clauses:
        for term in clause:
            if isinstance(term, ApplicationExpression) and term.function == answer_ex and (not isinstance(term.argument, IndividualVariableExpression)):
                answers.add(term.argument)
    return answers