import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def set_equal_probabilities(self):
    """Reset all probabilities to be an average value.

        Resets the values of all initial probabilities and all allowed
        transitions and all allowed emissions to be equal to 1 divided by the
        number of possible elements.

        This is useful if you just want to initialize a Markov Model to
        starting values (ie. if you have no prior notions of what the
        probabilities should be -- or if you are just feeling too lazy
        to calculate them :-).

        Warning 1 -- this will reset all currently set probabilities.

        Warning 2 -- This just sets all probabilities for transitions and
        emissions to total up to 1, so it doesn't ensure that the sum of
        each set of transitions adds up to 1.
        """
    new_initial_prob = 1.0 / len(self.transition_prob)
    for state in self._state_alphabet:
        self.initial_prob[state] = new_initial_prob
    new_trans_prob = 1.0 / len(self.transition_prob)
    for key in self.transition_prob:
        self.transition_prob[key] = new_trans_prob
    new_emission_prob = 1.0 / len(self.emission_prob)
    for key in self.emission_prob:
        self.emission_prob[key] = new_emission_prob