import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def set_initial_probabilities(self, initial_prob):
    """Set initial state probabilities.

        initial_prob is a dictionary mapping states to probabilities.
        Suppose, for example, that the state alphabet is ('A', 'B'). Call
        set_initial_prob({'A': 1}) to guarantee that the initial
        state will be 'A'. Call set_initial_prob({'A': 0.5, 'B': 0.5})
        to make each initial state equally probable.

        This method must now be called in order to use the Markov model
        because the calculation of initial probabilities has changed
        incompatibly; the previous calculation was incorrect.

        If initial probabilities are set for all states, then they should add up
        to 1. Otherwise the sum should be <= 1. The residual probability is
        divided up evenly between all the states for which the initial
        probability has not been set. For example, calling
        set_initial_prob({}) results in P('A') = 0.5 and P('B') = 0.5,
        for the above example.
        """
    self.initial_prob = copy.copy(initial_prob)
    for state in initial_prob:
        if state not in self._state_alphabet:
            raise ValueError(f'State {state} was not found in the sequence alphabet')
    num_states_not_set = len(self._state_alphabet) - len(self.initial_prob)
    if num_states_not_set < 0:
        raise Exception("Initial probabilities can't exceed # of states")
    prob_sum = sum(self.initial_prob.values())
    if prob_sum > 1.0:
        raise Exception('Total initial probability cannot exceed 1.0')
    if num_states_not_set > 0:
        prob = (1.0 - prob_sum) / num_states_not_set
        for state in self._state_alphabet:
            if state not in self.initial_prob:
                self.initial_prob[state] = prob