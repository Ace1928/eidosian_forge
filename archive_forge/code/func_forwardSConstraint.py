from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def forwardSConstraint(left, right):
    if not bothForward(left, right):
        return False
    return left.res().dir().is_forward() and left.arg().is_primitive()