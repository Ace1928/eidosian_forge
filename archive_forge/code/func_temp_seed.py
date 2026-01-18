import copy
import functools
import itertools
import multiprocessing.pool
import os
import queue
import re
import types
import warnings
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from multiprocessing import Manager
from queue import Empty
from shutil import disk_usage
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse
import multiprocess
import multiprocess.pool
import numpy as np
from tqdm.auto import tqdm
from .. import config
from ..parallel import parallel_map
from . import logging
from . import tqdm as hf_tqdm
from ._dill import (  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
@contextmanager
def temp_seed(seed: int, set_pytorch=False, set_tensorflow=False):
    """Temporarily set the random seed. This works for python numpy, pytorch and tensorflow."""
    np_state = np.random.get_state()
    np.random.seed(seed)
    if set_pytorch and config.TORCH_AVAILABLE:
        import torch
        torch_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch_cuda_states = torch.cuda.get_rng_state_all()
            torch.cuda.manual_seed_all(seed)
    if set_tensorflow and config.TF_AVAILABLE:
        import tensorflow as tf
        from tensorflow.python.eager import context as tfpycontext
        tf_state = tf.random.get_global_generator()
        temp_gen = tf.random.Generator.from_seed(seed)
        tf.random.set_global_generator(temp_gen)
        if not tf.executing_eagerly():
            raise ValueError('Setting random seed for TensorFlow is only available in eager mode')
        tf_context = tfpycontext.context()
        tf_seed = tf_context._seed
        tf_rng_initialized = hasattr(tf_context, '_rng')
        if tf_rng_initialized:
            tf_rng = tf_context._rng
        tf_context._set_global_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        if set_pytorch and config.TORCH_AVAILABLE:
            torch.random.set_rng_state(torch_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(torch_cuda_states)
        if set_tensorflow and config.TF_AVAILABLE:
            tf.random.set_global_generator(tf_state)
            tf_context._seed = tf_seed
            if tf_rng_initialized:
                tf_context._rng = tf_rng
            else:
                delattr(tf_context, '_rng')