import itertools
from typing import Dict, Iterator, List, Optional, Sequence, cast
import numpy as np
def get_outcome_probs(self) -> Dict[str, float]:
    """
        Parses a wavefunction (array of complex amplitudes) and returns a dictionary of
        outcomes and associated probabilities.

        :return: A dict with outcomes as keys and probabilities as values.
        :rtype: dict
        """
    outcome_dict = {}
    qubit_num = len(self)
    for index, amplitude in enumerate(self.amplitudes):
        outcome = get_bitstring_from_index(index, qubit_num)
        outcome_dict[outcome] = abs(amplitude) ** 2
    return outcome_dict