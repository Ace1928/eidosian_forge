import abc
import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntFlag
from multiprocessing import synchronize
from types import FrameType
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record
from torch.distributed.elastic.multiprocessing.redirects import (
from torch.distributed.elastic.multiprocessing.tail_log import TailLog
def get_std_cm(std_rd: str, redirect_fn):
    if IS_WINDOWS or IS_MACOS or (not std_rd):
        return nullcontext()
    else:
        return redirect_fn(std_rd)