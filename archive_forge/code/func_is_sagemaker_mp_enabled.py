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
def is_sagemaker_mp_enabled():
    smp_options = os.getenv('SM_HP_MP_PARAMETERS', '{}')
    try:
        smp_options = json.loads(smp_options)
        if 'partitions' not in smp_options:
            return False
    except json.JSONDecodeError:
        return False
    mpi_options = os.getenv('SM_FRAMEWORK_PARAMS', '{}')
    try:
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get('sagemaker_mpi_enabled', False):
            return False
    except json.JSONDecodeError:
        return False
    return _smdistributed_available