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
def peak_monitor_func(self):
    self.cpu_mem_used_peak = -1
    while True:
        self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)
        if not self.peak_monitoring:
            break