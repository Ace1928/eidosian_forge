import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def is_leftcorner(self, cat, left):
    """
        True if left is a leftcorner of cat, where left can be a
        terminal or a nonterminal.

        :param cat: the parent of the leftcorner
        :type cat: Nonterminal
        :param left: the suggested leftcorner
        :type left: Terminal or Nonterminal
        :rtype: bool
        """
    if is_nonterminal(left):
        return left in self.leftcorners(cat)
    elif self._leftcorner_words:
        return left in self._leftcorner_words.get(cat, set())
    else:
        return any((left in self._immediate_leftcorner_words.get(parent, set()) for parent in self.leftcorners(cat)))