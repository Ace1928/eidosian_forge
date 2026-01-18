import glob
import importlib
from os.path import basename, dirname, isfile, join
import torch
from torch._export.db.case import (
from . import *  # noqa: F403
def get_rewrite_cases(case):
    return _EXAMPLE_REWRITE_CASES.get(case.name, [])