import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def set_random_emission_probabilities(self):
    """Set all allowed emission probabilities to a randomly generated distribution.

        Returns the dictionary containing the emission probabilities.
        """
    if not self.emission_prob:
        raise Exception('No emissions have been allowed yet. Allow some or all emissions.')
    emissions = _calculate_emissions(self.emission_prob)
    for state in emissions:
        freqs = _gen_random_array(len(emissions[state]))
        for symbol in emissions[state]:
            self.emission_prob[state, symbol] = freqs.pop()
    return self.emission_prob