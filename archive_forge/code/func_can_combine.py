from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def can_combine(self, function, arg):
    if not (arg.is_function() and arg.res().is_function()):
        return False
    arg = innermostFunction(arg)
    subs = left.can_unify(arg_categ.arg())
    if subs is not None:
        return True
    return False