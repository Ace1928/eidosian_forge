import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def set_wandb_attrs(cbk, val_data):
    if isinstance(cbk, WandbCallback):
        if is_generator_like(val_data):
            cbk.generator = val_data
        elif is_dataset(val_data):
            if context.executing_eagerly():
                cbk.generator = iter(val_data)
            else:
                wandb.termwarn("Found a validation dataset in graph mode, can't patch Keras.")
        elif isinstance(val_data, tuple) and isinstance(val_data[0], tf.Tensor):

            def gen():
                while True:
                    yield K.get_session().run(val_data)
            cbk.generator = gen()
        else:
            cbk.validation_data = val_data