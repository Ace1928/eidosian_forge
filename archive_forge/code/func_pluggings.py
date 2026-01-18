from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def pluggings(self):
    """
        Calculate and return all the legal pluggings (mappings of labels to
        holes) of this semantics given the constraints.
        """
    record = []
    self._plug_nodes([(self.top_hole, [])], self.top_most_labels, {}, record)
    return record