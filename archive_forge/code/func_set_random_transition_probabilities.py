import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def set_random_transition_probabilities(self):
    """Set all allowed transition probabilities to a randomly generated distribution.

        Returns the dictionary containing the transition probabilities.
        """
    if not self.transition_prob:
        raise Exception('No transitions have been allowed yet. Allow some or all transitions by calling allow_transition or allow_all_transitions first.')
    transitions_from = _calculate_from_transitions(self.transition_prob)
    for from_state in transitions_from:
        freqs = _gen_random_array(len(transitions_from[from_state]))
        for to_state in transitions_from[from_state]:
            self.transition_prob[from_state, to_state] = freqs.pop()
    return self.transition_prob