import datetime
import io
import json
import os
import re
import tempfile
import threading
import warnings
import zipfile
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src import losses
from keras.src.engine import base_layer
from keras.src.optimizers import optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
def save_weights_only(model, filepath):
    """Save only the weights of a model to a target filepath (.weights.h5).

    Note: only supports h5 for now.
    """
    keras_saving_gauge.get_cell('save_weights_v3').set(True)
    filepath = str(filepath)
    if not filepath.endswith('.weights.h5'):
        raise ValueError(f'Invalid `filepath` argument: expected a `.weights.h5` extension. Received: filepath={filepath}')
    weights_store = H5IOStore(filepath, mode='w')
    _save_state(model, weights_store=weights_store, assets_store=None, inner_path='', visited_trackables=set())
    weights_store.close()