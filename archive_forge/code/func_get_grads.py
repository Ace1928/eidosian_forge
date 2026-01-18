from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import training_util
from tensorflow.python.util import nest
def get_grads(self, loss, params):
    return self.optimizer.compute_gradients(loss, params)