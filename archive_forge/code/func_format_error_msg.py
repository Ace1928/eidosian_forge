import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import cast, NoReturn, Optional
import torch._guards
from . import config
from .config import is_fbcode
from .utils import counters
import logging
def format_error_msg(exc: Exception, code, record_filename=None, frame=None) -> str:
    msg = os.linesep * 2
    if config.verbose:
        msg = format_error_msg_verbose(exc, code, record_filename, frame)
    else:
        msg = f"WON'T CONVERT {code.co_name} {code.co_filename} line {code.co_firstlineno} \ndue to: \n{format_exc()}"
    return msg