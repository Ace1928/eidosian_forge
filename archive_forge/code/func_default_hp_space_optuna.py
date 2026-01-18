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
def default_hp_space_optuna(trial) -> Dict[str, float]:
    from .integrations import is_optuna_available
    assert is_optuna_available(), 'This function needs Optuna installed: `pip install optuna`'
    return {'learning_rate': trial.suggest_float('learning_rate', 1e-06, 0.0001, log=True), 'num_train_epochs': trial.suggest_int('num_train_epochs', 1, 5), 'seed': trial.suggest_int('seed', 1, 40), 'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16, 32, 64])}