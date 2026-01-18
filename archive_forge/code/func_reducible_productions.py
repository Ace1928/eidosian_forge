from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import Tree
def reducible_productions(self):
    """
        :return: A list of the productions for which reductions are
            available for the current parser state.
        :rtype: list(Production)
        """
    productions = []
    for production in self._grammar.productions():
        rhslen = len(production.rhs())
        if self._match_rhs(production.rhs(), self._stack[-rhslen:]):
            productions.append(production)
    return productions