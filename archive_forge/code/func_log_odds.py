import math
import numbers
import numpy as np
from Bio.Seq import Seq
from . import _pwm  # type: ignore
def log_odds(self, background=None):
    """Return the Position-Specific Scoring Matrix.

        The Position-Specific Scoring Matrix (PSSM) contains the log-odds
        scores computed from the probability matrix and the background
        probabilities. If the background is None, a uniform background
        distribution is assumed.
        """
    values = {}
    alphabet = self.alphabet
    if background is None:
        background = dict.fromkeys(self.alphabet, 1.0)
    else:
        background = dict(background)
    total = sum(background.values())
    for letter in alphabet:
        background[letter] /= total
        values[letter] = []
    for i in range(self.length):
        for letter in alphabet:
            b = background[letter]
            if b > 0:
                p = self[letter][i]
                if p > 0:
                    logodds = math.log(p / b, 2)
                else:
                    logodds = -math.inf
            else:
                p = self[letter][i]
                if p > 0:
                    logodds = math.inf
                else:
                    logodds = math.nan
            values[letter].append(logodds)
    pssm = PositionSpecificScoringMatrix(alphabet, values)
    return pssm