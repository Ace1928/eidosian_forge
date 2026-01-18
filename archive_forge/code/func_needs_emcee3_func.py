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
def needs_emcee3_func():
    """Check if emcee3 is required."""
    needs_emcee3 = pytest.mark.skipif(emcee_version() < 3, reason='emcee3 required')
    return needs_emcee3