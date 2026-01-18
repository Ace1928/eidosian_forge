import os
import re
import shutil
import tempfile
from datetime import datetime
from time import time
from typing import List, Optional
from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, Method
from cmdstanpy.utils import get_logger
def raise_for_timeouts(self) -> None:
    if any(self._timeout_flags):
        raise TimeoutError(f'{sum(self._timeout_flags)} of {self.num_procs} processes timed out')