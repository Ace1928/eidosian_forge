from __future__ import generators
from bisect import bisect_right
import sys
import inspect, tokenize
import py
from types import ModuleType
def readline_generator(lines):
    for line in lines:
        yield (line + '\n')
    while True:
        yield ''