import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
def derive_stage(self):
    """derives the stage/caller name automatically"""
    caller = inspect.currentframe().f_back.f_back.f_code.co_name
    if caller in self.stages:
        return self.stages[caller]
    else:
        raise ValueError(f'was called from {caller}, but only expect to be called from one of {self.stages.keys()}')