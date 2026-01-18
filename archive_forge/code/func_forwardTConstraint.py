from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def forwardTConstraint(left, right):
    arg = innermostFunction(right)
    return arg.dir().is_backward() and arg.res().is_primitive()