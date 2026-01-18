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
def get_major_and_minor_from_version(full_version):
    return str(version.parse(full_version).major) + '.' + str(version.parse(full_version).minor)