import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def set_transition_pseudocount(self, from_state, to_state, count):
    """Set the default pseudocount for a transition.

        To avoid computational problems, it is helpful to be able to
        set a 'default' pseudocount to start with for estimating
        transition and emission probabilities (see p62 in Durbin et al
        for more discussion on this. By default, all transitions have
        a pseudocount of 1.

        Raises:
        KeyError if the transition is not allowed.

        """
    if (from_state, to_state) in self.transition_pseudo:
        self.transition_pseudo[from_state, to_state] = count
    else:
        raise KeyError(f'Transition from {from_state} to {to_state} is not allowed.')