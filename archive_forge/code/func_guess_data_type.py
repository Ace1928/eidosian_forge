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
def guess_data_type(shape: Sequence[int], risky: bool=False) -> Optional[str]:
    """Infer the type of data based on the shape of the tensors.

    Arguments:
        shape (Sequence[int]): The shape of the data
        risky(bool): some guesses are more likely to be wrong.
    """
    if len(shape) in (1, 2):
        return 'label'
    if risky and len(shape) == 3:
        return 'image'
    if len(shape) == 4:
        if shape[-1] in (1, 3, 4):
            return 'image'
        else:
            return 'segmentation_mask'
    return None