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
def default_hp_space_ray(trial) -> Dict[str, float]:
    from .integrations import is_ray_tune_available
    assert is_ray_tune_available(), 'This function needs ray installed: `pip install ray[tune]`'
    from ray import tune
    return {'learning_rate': tune.loguniform(1e-06, 0.0001), 'num_train_epochs': tune.choice(list(range(1, 6))), 'seed': tune.uniform(1, 40), 'per_device_train_batch_size': tune.choice([4, 8, 16, 32, 64])}