from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def forwardOnly(left, right):
    return left.dir().is_forward()