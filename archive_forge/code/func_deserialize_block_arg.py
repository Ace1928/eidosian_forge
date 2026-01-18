import json
import multiprocessing
import os
from typing import Optional
from typing import Tuple
import numpy as np
import tensorflow as tf
from keras_tuner.engine import hyperparameters
def deserialize_block_arg(arg):
    if isinstance(arg, dict):
        return hyperparameters.deserialize(arg)
    return arg