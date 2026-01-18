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
def default_hp_space_sigopt(trial):
    return [{'bounds': {'min': 1e-06, 'max': 0.0001}, 'name': 'learning_rate', 'type': 'double', 'transformamtion': 'log'}, {'bounds': {'min': 1, 'max': 6}, 'name': 'num_train_epochs', 'type': 'int'}, {'bounds': {'min': 1, 'max': 40}, 'name': 'seed', 'type': 'int'}, {'categorical_values': ['4', '8', '16', '32', '64'], 'name': 'per_device_train_batch_size', 'type': 'categorical'}]