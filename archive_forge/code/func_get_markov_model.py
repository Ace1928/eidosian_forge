import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def get_markov_model(self):
    """Return the markov model corresponding with the current parameters.

        Each markov model returned by a call to this function is unique
        (ie. they don't influence each other).
        """
    if not self.initial_prob:
        raise Exception('set_initial_probabilities must be called to fully initialize the Markov model')
    initial_prob = copy.deepcopy(self.initial_prob)
    transition_prob = copy.deepcopy(self.transition_prob)
    emission_prob = copy.deepcopy(self.emission_prob)
    transition_pseudo = copy.deepcopy(self.transition_pseudo)
    emission_pseudo = copy.deepcopy(self.emission_pseudo)
    return HiddenMarkovModel(self._state_alphabet, self._emission_alphabet, initial_prob, transition_prob, emission_prob, transition_pseudo, emission_pseudo)