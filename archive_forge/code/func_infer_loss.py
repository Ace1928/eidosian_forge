from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from autokeras import adapters
from autokeras import analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras import preprocessors
from autokeras.blocks import reduction
from autokeras.engine import head as head_module
from autokeras.utils import types
from autokeras.utils import utils
def infer_loss(self):
    if not self.num_classes:
        return None
    if self.num_classes == 2 or self.multi_label:
        return losses.BinaryCrossentropy()
    return losses.CategoricalCrossentropy()