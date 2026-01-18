import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
def emcee_version():
    """Check emcee version.

    Returns
    -------
    int
        Major version number

    """
    import emcee
    return int(emcee.__version__[0])