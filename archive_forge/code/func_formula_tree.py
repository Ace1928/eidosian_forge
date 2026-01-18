from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def formula_tree(self, plugging):
    """
        Return the first-order logic formula tree for this underspecified
        representation using the plugging given.
        """
    return self._formula_tree(plugging, self.top_hole)