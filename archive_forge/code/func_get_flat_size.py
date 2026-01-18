from collections import OrderedDict, deque
import numpy as np
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_tf
def get_flat_size(self):
    """Returns the total length of all of the flattened variables.

        Returns:
            The length of all flattened variables concatenated.
        """
    return sum((np.prod(v.get_shape().as_list()) for v in self.variables.values()))