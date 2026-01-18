from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def currently_complete(self):
    """
        :return: Whether the parser's current state represents a
            complete parse.
        :rtype: bool
        """
    return len(self._frontier) == 0 and len(self._rtext) == 0