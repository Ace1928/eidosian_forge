import collections
import copy
import itertools
import random
import re
import warnings
@property
def clade(self):
    """Return first clade in this tree (not itself)."""
    return self.root