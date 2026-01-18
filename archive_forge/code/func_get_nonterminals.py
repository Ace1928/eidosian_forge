import collections
import copy
import itertools
import random
import re
import warnings
def get_nonterminals(self, order='preorder'):
    """Get a list of all of this tree's nonterminal (internal) nodes."""
    return list(self.find_clades(terminal=False, order=order))