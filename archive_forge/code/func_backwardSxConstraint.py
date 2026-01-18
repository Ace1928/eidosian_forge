from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def backwardSxConstraint(left, right):
    if not left.dir().can_cross() and right.dir().can_cross():
        return False
    if not bothForward(left, right):
        return False
    return right.res().dir().is_backward() and right.arg().is_primitive()