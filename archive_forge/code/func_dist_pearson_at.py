import math
import numbers
import numpy as np
from Bio.Seq import Seq
from . import _pwm  # type: ignore
def dist_pearson_at(self, other, offset):
    """Return the similarity score based on pearson correlation at the given offset."""
    letters = self.alphabet
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    norm = max(self.length, offset + other.length) * len(letters)
    for pos in range(min(self.length - offset, other.length)):
        xi = [self[letter, pos + offset] for letter in letters]
        yi = [other[letter, pos] for letter in letters]
        sx += sum(xi)
        sy += sum(yi)
        sxx += sum((x * x for x in xi))
        sxy += sum((x * y for x, y in zip(xi, yi)))
        syy += sum((y * y for y in yi))
    sx /= norm
    sy /= norm
    sxx /= norm
    sxy /= norm
    syy /= norm
    numerator = sxy - sx * sy
    denominator = math.sqrt((sxx - sx * sx) * (syy - sy * sy))
    return numerator / denominator