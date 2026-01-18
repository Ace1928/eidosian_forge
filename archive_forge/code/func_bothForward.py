from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def bothForward(left, right):
    return left.dir().is_forward() and right.dir().is_forward()