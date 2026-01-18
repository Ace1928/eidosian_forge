import logging
import math
import os
import sys
from taskflow import engines
from taskflow.engines.worker_based import worker
from taskflow.patterns import unordered_flow as uf
from taskflow import task
from taskflow.utils import threading_utils
def mandelbrot(x, y, max_iters):
    c = complex(x, y)
    z = 0j
    for i in range(max_iters):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return i
    return max_iters