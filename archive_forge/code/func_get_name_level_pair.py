import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
import torch._guards
import torch.distributed as dist
def get_name_level_pair(name):
    clean_name = name.replace(INCR_VERBOSITY_CHAR, '')
    clean_name = clean_name.replace(DECR_VERBOSITY_CHAR, '')
    if name[0] == INCR_VERBOSITY_CHAR:
        level = logging.DEBUG
    elif name[0] == DECR_VERBOSITY_CHAR:
        level = logging.ERROR
    else:
        level = logging.INFO
    return (clean_name, level)