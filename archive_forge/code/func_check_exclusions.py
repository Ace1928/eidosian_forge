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
@_pytest_fn_decorator
def check_exclusions(fn, *args, **kw):
    _exclusions = args[-1]
    if _exclusions:
        exlu = exclusions.compound().add(*_exclusions)
        fn = exlu(fn)
    return fn(*args[:-1], **kw)