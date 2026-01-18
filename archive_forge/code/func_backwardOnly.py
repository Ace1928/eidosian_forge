from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def backwardOnly(left, right):
    return right.dir().is_backward()