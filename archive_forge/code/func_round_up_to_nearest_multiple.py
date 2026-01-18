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
def round_up_to_nearest_multiple(n: int, m: int) -> int:
    return m * ((n + m - 1) // m)