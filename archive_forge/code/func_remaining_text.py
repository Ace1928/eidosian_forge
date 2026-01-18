from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def remaining_text(self):
    """
        :return: The portion of the text that is not yet covered by the
            tree.
        :rtype: list(str)
        """
    return self._rtext