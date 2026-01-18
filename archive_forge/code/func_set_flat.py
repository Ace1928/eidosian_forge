from collections import OrderedDict, deque
import numpy as np
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_tf
def set_flat(self, new_weights):
    """Sets the weights to new_weights, converting from a flat array.

        Note:
            You can only set all weights in the network using this function,
            i.e., the length of the array must match get_flat_size.

        Args:
            new_weights (np.ndarray): Flat array containing weights.
        """
    shapes = [v.get_shape().as_list() for v in self.variables.values()]
    arrays = unflatten(new_weights, shapes)
    if not self.sess:
        for v, a in zip(self.variables.values(), arrays):
            v.assign(a)
    else:
        placeholders = [self.placeholders[k] for k, v in self.variables.items()]
        self.sess.run(list(self.assignment_nodes.values()), feed_dict=dict(zip(placeholders, arrays)))