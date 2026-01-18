import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def allow_transition(self, from_state, to_state, probability=None, pseudocount=None):
    """Set a transition as being possible between the two states.

        probability and pseudocount are optional arguments
        specifying the probabilities and pseudo counts for the transition.
        If these are not supplied, then the values are set to the
        default values.

        Raises:
        KeyError -- if the two states already have an allowed transition.

        """
    for state in [from_state, to_state]:
        if state not in self._state_alphabet:
            raise ValueError(f'State {state} was not found in the sequence alphabet')
    if (from_state, to_state) not in self.transition_prob and (from_state, to_state) not in self.transition_pseudo:
        if probability is None:
            probability = 0
        self.transition_prob[from_state, to_state] = probability
        if pseudocount is None:
            pseudocount = self.DEFAULT_PSEUDO
        self.transition_pseudo[from_state, to_state] = pseudocount
    else:
        raise KeyError(f'Transition from {from_state} to {to_state} is already allowed.')