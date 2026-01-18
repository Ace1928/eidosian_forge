from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
@classmethod
def get_instance(cls):
    if cls._instance is None:
        cls._instance = cls.__new__(cls)
        cls._instance.initialize()
    return cls._instance