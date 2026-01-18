import re
import warnings
import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name
def scale_loss(self, loss):
    """Scale the loss before computing gradients.

        Scales the loss before gradients are computed in a `train_step`. This
        is primarily useful during mixed precision training to prevent numeric
        underflow.
        """
    if self.loss_scale_factor is not None:
        return loss * self.loss_scale_factor
    return loss