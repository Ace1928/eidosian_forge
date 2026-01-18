import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def sample_with_exponential_decay_weights(xs: Union[Iterable, Iterable[Iterable]], ys: Iterable[Iterable], keys: Optional[Iterable]=None, sample_size: int=1500) -> Tuple[List, List, Optional[List]]:
    """Sample from a list of lists with weights that decay exponentially.

    May be used with the wandb.plot.line_series function.
    """
    xs_array = np.array(xs)
    ys_array = np.array(ys)
    keys_array = np.array(keys) if keys else None
    weights = np.exp(-np.arange(len(xs_array)) / len(xs_array))
    weights /= np.sum(weights)
    sampled_indices = np.random.choice(len(xs_array), size=sample_size, p=weights)
    sampled_xs = xs_array[sampled_indices].tolist()
    sampled_ys = ys_array[sampled_indices].tolist()
    sampled_keys = keys_array[sampled_indices].tolist() if keys else None
    return (sampled_xs, sampled_ys, sampled_keys)