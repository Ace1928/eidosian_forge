from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def innermostFunction(categ):
    while categ.res().is_function():
        categ = categ.res()
    return categ