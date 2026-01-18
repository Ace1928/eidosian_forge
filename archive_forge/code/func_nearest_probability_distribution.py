from math import sqrt
import re
from .probability import ProbDistribution
def nearest_probability_distribution(self, return_distance=False):
    """Takes a quasiprobability distribution and maps
        it to the closest probability distribution as defined by
        the L2-norm.

        Parameters:
            return_distance (bool): Return the L2 distance between distributions.

        Returns:
            ProbDistribution: Nearest probability distribution.
            float: Euclidean (L2) distance of distributions.

        Notes:
            Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
        """
    sorted_probs = dict(sorted(self.items(), key=lambda item: item[1]))
    num_elems = len(sorted_probs)
    new_probs = {}
    beta = 0
    diff = 0
    for key, val in sorted_probs.items():
        temp = val + beta / num_elems
        if temp < 0:
            beta += val
            num_elems -= 1
            diff += val * val
        else:
            diff += beta / num_elems * (beta / num_elems)
            new_probs[key] = sorted_probs[key] + beta / num_elems
    if return_distance:
        return (ProbDistribution(new_probs, self.shots), sqrt(diff))
    return ProbDistribution(new_probs, self.shots)