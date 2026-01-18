import math
import numbers
import numpy as np
from Bio.Seq import Seq
from . import _pwm  # type: ignore
def dist_pearson(self, other):
    """Return the similarity score based on pearson correlation for the given motif against self.

        We use the Pearson's correlation of the respective probabilities.
        """
    if self.alphabet != other.alphabet:
        raise ValueError('Cannot compare motifs with different alphabets')
    max_p = -2
    for offset in range(-self.length + 1, other.length):
        if offset < 0:
            p = self.dist_pearson_at(other, -offset)
        else:
            p = other.dist_pearson_at(self, offset)
        if max_p < p:
            max_p = p
            max_o = -offset
    return (1 - max_p, max_o)