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
@property
def num_runs(self) -> int:
    return len(self.latencies)