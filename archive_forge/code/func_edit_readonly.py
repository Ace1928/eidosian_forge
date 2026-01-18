from __future__ import annotations
import inspect
from contextlib import contextmanager
from typing import Any, Dict, Iterator
import param
from packaging.version import Version
@contextmanager
def edit_readonly(parameterized: param.Parameterized) -> Iterator:
    """
    Temporarily set parameters on Parameterized object to readonly=False
    to allow editing them.
    """
    params = parameterized.param.objects('existing').values()
    readonlys = [p.readonly for p in params]
    constants = [p.constant for p in params]
    for p in params:
        p.readonly = False
        p.constant = False
    try:
        yield
    except Exception:
        raise
    finally:
        for p, readonly in zip(params, readonlys):
            p.readonly = readonly
        for p, constant in zip(params, constants):
            p.constant = constant