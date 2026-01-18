import os
import platform
import subprocess
import sys
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from .. import progress as progbar
from .logging import get_logger
def set_cmdstan_path(path: str) -> None:
    """
    Validate, then set CmdStan directory path.
    """
    validate_cmdstan_path(path)
    os.environ['CMDSTAN'] = path