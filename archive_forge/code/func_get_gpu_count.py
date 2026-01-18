import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch
import urllib3
from transformers import logging as transformers_logging
from .integrations import (
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
import asyncio  # noqa
def get_gpu_count():
    """
    Return the number of available gpus (regardless of whether torch, tf or jax is used)
    """
    if is_torch_available():
        import torch
        return torch.cuda.device_count()
    elif is_tf_available():
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU'))
    elif is_flax_available():
        import jax
        return jax.device_count()
    else:
        return 0