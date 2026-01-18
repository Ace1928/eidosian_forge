import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
def is_torchdynamo_compiling():
    if not is_torch_available():
        return False
    try:
        import torch._dynamo as dynamo
        return dynamo.is_compiling()
    except Exception:
        return False