import math
import numpy as np
from tensorflow.python.ops.distributions import special_math
def normal_cdf(x):
    """Cumulative distribution function for a standard normal distribution."""
    return 0.5 + 0.5 * np.vectorize(math.erf)(x / math.sqrt(2))