import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def set_random_probabilities(self):
    """Set all probabilities to randomly generated numbers.

        Resets probabilities of all initial states, transitions, and
        emissions to random values.
        """
    self.set_random_initial_probabilities()
    self.set_random_transition_probabilities()
    self.set_random_emission_probabilities()