import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
def enable_full_determinism(seed: int, warn_only: bool=False):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    set_seed(seed)
    if is_torch_available():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if is_tf_available():
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()