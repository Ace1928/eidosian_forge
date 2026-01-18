import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
def no_dipy():
    """Check if dipy is available."""
    global HAVE_DIPY
    return not HAVE_DIPY