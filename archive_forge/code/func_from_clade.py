import collections
import copy
import itertools
import random
import re
import warnings
@classmethod
def from_clade(cls, clade, **kwargs):
    """Create a new Tree object given a clade.

        Keyword arguments are the usual ``Tree`` constructor parameters.
        """
    root = copy.deepcopy(clade)
    return cls(root, **kwargs)