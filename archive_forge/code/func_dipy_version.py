import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
def dipy_version():
    """Check dipy version."""
    if no_dipy():
        return None
    return dipy.__version__