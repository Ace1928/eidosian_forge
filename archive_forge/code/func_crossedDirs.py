from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def crossedDirs(left, right):
    return left.dir().is_forward() and right.dir().is_backward()