import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def set_transition_score(self, from_state, to_state, probability):
    """Set the probability of a transition between two states.

        Raises:
        KeyError if the transition is not allowed.

        """
    if (from_state, to_state) in self.transition_prob:
        self.transition_prob[from_state, to_state] = probability
    else:
        raise KeyError(f'Transition from {from_state} to {to_state} is not allowed.')