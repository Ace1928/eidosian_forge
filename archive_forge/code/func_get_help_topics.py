import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def get_help_topics(self) -> List[str]:
    """Return a list of help topics"""
    all_topics = [name[len(constants.HELP_FUNC_PREFIX):] for name in self.get_names() if name.startswith(constants.HELP_FUNC_PREFIX) and callable(getattr(self, name))]
    return [topic for topic in all_topics if topic not in self.hidden_commands and topic not in self.disabled_commands]