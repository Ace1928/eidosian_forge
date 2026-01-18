import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def get_blank_transitions(self):
    """Get the default transitions for the model.

        Returns a dictionary of all of the default transitions between any
        two letters in the sequence alphabet. The dictionary is structured
        with keys as (letter1, letter2) and values as the starting number
        of transitions.
        """
    return self._transition_pseudo