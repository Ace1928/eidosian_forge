import importlib.util
import inspect
import sys
from collections import OrderedDict
from contextlib import contextmanager
from typing import Tuple, Union
import numpy as np
from packaging import version
from transformers.utils import is_torch_available
def is_sentence_transformers_available():
    return _sentence_transformers_available