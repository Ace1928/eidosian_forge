import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def get_BooleanExpression_factory(self, tok):
    """This method serves as a hook for other logic parsers that
        have different boolean operators"""
    if tok == DrtTokens.DRS_CONC:
        return lambda first, second: DrtConcatenation(first, second, None)
    elif tok in DrtTokens.OR_LIST:
        return DrtOrExpression
    elif tok in DrtTokens.IMP_LIST:

        def make_imp_expression(first, second):
            if isinstance(first, DRS):
                return DRS(first.refs, first.conds, second)
            if isinstance(first, DrtConcatenation):
                return DrtConcatenation(first.first, first.second, second)
            raise Exception('Antecedent of implication must be a DRS')
        return make_imp_expression
    else:
        return None