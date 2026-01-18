import re
from collections import defaultdict
from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression
def parsePrimitiveCategory(chunks, primitives, families, var):
    """
    Parse a primitive category

    If the primitive is the special category 'var', replace it with the
    correct `CCGVar`.
    """
    if chunks[0] == 'var':
        if chunks[1] is None:
            if var is None:
                var = CCGVar()
            return (var, var)
    catstr = chunks[0]
    if catstr in families:
        cat, cvar = families[catstr]
        if var is None:
            var = cvar
        else:
            cat = cat.substitute([(cvar, var)])
        return (cat, var)
    if catstr in primitives:
        subscrs = parseSubscripts(chunks[1])
        return (PrimitiveCategory(catstr, subscrs), var)
    raise AssertionError("String '" + catstr + "' is neither a family nor primitive category.")