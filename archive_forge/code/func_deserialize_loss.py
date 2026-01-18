from typing import Optional
from tensorflow import keras
from autokeras.engine import io_hypermodel
from autokeras.utils import types
def deserialize_loss(loss):
    if isinstance(loss, list):
        return loss[0]
    return keras.losses.deserialize(loss)