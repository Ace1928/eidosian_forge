import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
@property
def rejection_rate(self):
    if not self._total_generated:
        return 0.0
    return self._rejections / self._total_generated