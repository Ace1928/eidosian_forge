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
def cpu_mem_used(self):
    """get resident set size memory for the current process"""
    return self.process.memory_info().rss