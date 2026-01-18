import collections
import dataclasses
import importlib.metadata
import inspect
import logging
import multiprocessing
import os
import sys
import traceback
import types
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
from importlib.machinery import ModuleSpec
from unittest import mock
import duet
import numpy as np
import pandas as pd
import pytest
import sympy
from _pytest.outcomes import Failed
import cirq.testing
from cirq._compat import (
def test_deprecated_module_deadline_validation():
    with pytest.raises(AssertionError, match='deadline should match vX.Y'):
        deprecated_submodule(new_module_name='new', old_parent='old_p', old_child='old_ch', deadline='invalid', create_attribute=False)