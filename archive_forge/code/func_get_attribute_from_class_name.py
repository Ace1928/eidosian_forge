import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def get_attribute_from_class_name(class_name: str) -> Any:
    """Get Python attribute from the provided class name.

    The caller needs to make sure the provided class name includes
    full module name, and can be imported successfully.
    """
    from importlib import import_module
    paths = class_name.split('.')
    if len(paths) < 2:
        raise ValueError(f'Cannot create object from {class_name}.')
    module_name = '.'.join(paths[:-1])
    attribute_name = paths[-1]
    return getattr(import_module(module_name), attribute_name)