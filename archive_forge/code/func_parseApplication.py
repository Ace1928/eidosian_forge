import re
from collections import defaultdict
from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression
def parseApplication(app):
    """
    Parse an application operator
    """
    return Direction(app[0], app[1:])