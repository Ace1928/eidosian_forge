from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
def get_num_samples(self, opt) -> Tuple[int, int]:
    datatype = opt['datatype']
    if 'train' in datatype:
        return (INFINITE, INFINITE)
    elif 'valid' in datatype:
        return (NUM_TEST, NUM_TEST)
    elif 'test' in datatype:
        return (NUM_TEST, NUM_TEST)