import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def allow_all_transitions(self):
    """Create transitions between all states.

        By default all transitions within the alphabet are disallowed;
        this is a convenience function to change this to allow all
        possible transitions.
        """
    all_probs = self._all_blank(self._state_alphabet, self._state_alphabet)
    all_pseudo = self._all_pseudo(self._state_alphabet, self._state_alphabet)
    for set_key in self.transition_prob:
        all_probs[set_key] = self.transition_prob[set_key]
    for set_key in self.transition_pseudo:
        all_pseudo[set_key] = self.transition_pseudo[set_key]
    self.transition_prob = all_probs
    self.transition_pseudo = all_pseudo