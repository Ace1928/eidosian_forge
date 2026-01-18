import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_init_multi_one_hot_encode():
    layer_module.MultiCategoryEncoding(encoding=[layer_module.ONE_HOT, layer_module.INT, layer_module.NONE])