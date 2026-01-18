import os
import sys
import signal
import threading
import logging
from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Any, Dict, List
from .mp_utils import _CPU_CORES, _MAX_THREADS, _MAX_PROCS
@property
def killed(self):
    return not IsLazyAlive