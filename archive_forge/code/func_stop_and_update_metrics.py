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
def stop_and_update_metrics(self, metrics=None):
    """combine stop and metrics update in one call for simpler code"""
    if self.skip_memory_metrics:
        return
    stage = self.derive_stage()
    self.stop(stage)
    if metrics is not None:
        self.update_metrics(stage, metrics)