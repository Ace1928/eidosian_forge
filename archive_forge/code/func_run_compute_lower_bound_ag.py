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
def run_compute_lower_bound_ag():
    for w, go in zip(weights, gathered_outputs):
        torch.matmul(gathered_input, w, out=go)