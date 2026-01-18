import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def satisfiers(self, parsed, varex, g, trace=None, nesting=0):
    """
        Generate the entities from the model's domain that satisfy an open formula.

        :param parsed: an open formula
        :type parsed: Expression
        :param varex: the relevant free individual variable in ``parsed``.
        :type varex: VariableExpression or str
        :param g: a variable assignment
        :type g:  Assignment
        :return: a set of the entities that satisfy ``parsed``.
        """
    spacer = '   '
    indent = spacer + spacer * nesting
    candidates = []
    if isinstance(varex, str):
        var = Variable(varex)
    else:
        var = varex
    if var in parsed.free():
        if trace:
            print()
            print(spacer * nesting + f"Open formula is '{parsed}' with assignment {g}")
        for u in self.domain:
            new_g = g.copy()
            new_g.add(var.name, u)
            if trace and trace > 1:
                lowtrace = trace - 1
            else:
                lowtrace = 0
            value = self.satisfy(parsed, new_g, lowtrace)
            if trace:
                print(indent + '(trying assignment %s)' % new_g)
            if value == False:
                if trace:
                    print(indent + f"value of '{parsed}' under {new_g} is False")
            else:
                candidates.append(u)
                if trace:
                    print(indent + f"value of '{parsed}' under {new_g} is {value}")
        result = {c for c in candidates}
    else:
        raise Undefined(f'{var.name} is not free in {parsed}')
    return result