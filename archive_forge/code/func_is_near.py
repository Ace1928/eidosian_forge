import collections
import math
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow
from taskflow import task
def is_near(val, expected, tolerance=0.001):
    if val > expected + tolerance:
        return False
    if val < expected - tolerance:
        return False
    return True