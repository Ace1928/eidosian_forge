import re
from collections import defaultdict
from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression
def matchBrackets(string):
    """
    Separate the contents matching the first set of brackets from the rest of
    the input.
    """
    rest = string[1:]
    inside = '('
    while rest != '' and (not rest.startswith(')')):
        if rest.startswith('('):
            part, rest = matchBrackets(rest)
            inside = inside + part
        else:
            inside = inside + rest[0]
            rest = rest[1:]
    if rest.startswith(')'):
        return (inside + ')', rest[1:])
    raise AssertionError("Unmatched bracket in string '" + string + "'")