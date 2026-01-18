from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def frontier(self):
    """
        :return: A list of the tree locations of all subtrees that
            have not yet been expanded, and all leaves that have not
            yet been matched.
        :rtype: list(tuple(int))
        """
    return self._frontier