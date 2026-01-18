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
def pystan_version():
    """Check PyStan version.

    Returns
    -------
    int
        Major version number

    """
    try:
        import pystan
        version = int(pystan.__version__[0])
    except ImportError:
        try:
            import stan
            version = int(stan.__version__[0])
        except ImportError:
            version = None
    return version