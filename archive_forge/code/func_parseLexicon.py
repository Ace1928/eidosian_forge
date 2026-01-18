import re
from collections import defaultdict
from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression
@deprecated('Use fromstring() instead.')
def parseLexicon(lex_str):
    return fromstring(lex_str)