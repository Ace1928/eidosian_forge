from __future__ import annotations
import inspect
from contextlib import contextmanager
from typing import Any, Dict, Iterator
import param
from packaging.version import Version
def get_params_to_inherit(parameterized: param.Parameterized) -> Dict:
    return {p: v for p, v in parameterized.param.values().items() if should_inherit(parameterized, p, v)}