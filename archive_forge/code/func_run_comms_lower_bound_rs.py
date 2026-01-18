import argparse
import contextlib
import dataclasses
import enum
import multiprocessing
import os
import random
from collections import deque
from statistics import mean, stdev
from typing import Callable
import torch
def run_comms_lower_bound_rs():
    for so, go in zip(scattered_outputs, gathered_outputs):
        torch.distributed.reduce_scatter_tensor(so, go)