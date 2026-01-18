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
def load_weights_only(model, filepath, skip_mismatch=False):
    """Load the weights of a model from a filepath (.keras or .weights.h5).

    Note: only supports h5 for now.
    """
    temp_dir = None
    archive = None
    filepath = str(filepath)
    if filepath.endswith('.weights.h5'):
        weights_store = H5IOStore(filepath, mode='r')
    elif filepath.endswith('.keras'):
        archive = zipfile.ZipFile(filepath, 'r')
        weights_store = H5IOStore(_VARS_FNAME + '.h5', archive=archive, mode='r')
    _load_state(model, weights_store=weights_store, assets_store=None, inner_path='', skip_mismatch=skip_mismatch, visited_trackables=set())
    weights_store.close()
    if temp_dir and tf.io.gfile.exists(temp_dir):
        tf.io.gfile.rmtree(temp_dir)
    if archive:
        archive.close()