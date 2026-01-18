import logging
import numpy as np
from .base import mx_real_t
from . import ndarray as nd
from .context import cpu
from .io import DataDesc
@property
def grad_arrays(self):
    """Shared gradient arrays."""
    return self.execgrp.grad_arrays