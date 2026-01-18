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
def llama_07B_MHA(world_size: int) -> Scenario:
    batch_size = 8
    return Scenario(num_samples=batch_size * LLAMA_07B_SLEN, outer_dim=LLAMA_07B_D, inner_dim=LLAMA_07B_D // world_size, num_ag_matrices=3)