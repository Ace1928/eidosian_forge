from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def pytest_pycollect_makeitem(collector, name, obj):
    if inspect.isclass(obj) and plugin_base.want_class(name, obj):
        from sqlalchemy.testing import config
        if config.any_async:
            obj = _apply_maybe_async(obj)
        return [pytest.Class.from_parent(name=parametrize_cls.__name__, parent=collector) for parametrize_cls in _parametrize_cls(collector.module, obj)]
    elif inspect.isfunction(obj) and collector.cls is not None and plugin_base.want_method(collector.cls, obj):
        return None
    else:
        return []