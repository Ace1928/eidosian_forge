import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def output_can_be_silenced(magic_func):
    """Mark a magic function so its output may be silenced.

    The output is silenced if the Python code used as a parameter of
    the magic ends in a semicolon, not counting a Python comment that can
    follow it.
    """
    setattr(magic_func, MAGIC_OUTPUT_CAN_BE_SILENCED, True)
    return magic_func