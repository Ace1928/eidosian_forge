from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def bothBackward(left, right):
    return left.dir().is_backward() and right.dir().is_backward()