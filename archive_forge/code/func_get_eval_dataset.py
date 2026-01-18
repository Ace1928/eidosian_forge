import os
import subprocess
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Set
import numpy as np
import optuna
import torch
import transformers
from datasets import Dataset
from tqdm import trange
from . import version as optimum_version
from .utils.preprocessing import (
from .utils.runs import RunConfig, cpu_info_command
def get_eval_dataset(self):
    """
        Get evaluation dataset.  The dataset needs to be loaded first with [`~optimum.runs_base.Run.load_datasets`].

        Returns:
            `datasets.Dataset`: Evaluation dataset.
        """
    if not hasattr(self, '_eval_dataset'):
        raise KeyError('No evaluation dataset defined for this run.')
    return self._eval_dataset