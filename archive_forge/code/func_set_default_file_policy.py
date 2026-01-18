import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
def set_default_file_policy(self, filename: str, file_policy: 'DefaultFilePolicy') -> None:
    """Set an upload policy for a file unless one has already been set."""
    if filename not in self._file_policies:
        self._file_policies[filename] = file_policy