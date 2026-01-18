import copyreg
import io
import functools
import types
import sys
import os
from multiprocessing import util
from pickle import loads, HIGHEST_PROTOCOL
def get_loky_pickler():
    global _LokyPickler
    return _LokyPickler