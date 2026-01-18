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
def find_commandsets(self, commandset_type: Type[CommandSet], *, subclass_match: bool=False) -> List[CommandSet]:
    """
        Find all CommandSets that match the provided CommandSet type.
        By default, locates a CommandSet that is an exact type match but may optionally return all CommandSets that
        are sub-classes of the provided type
        :param commandset_type: CommandSet sub-class type to search for
        :param subclass_match: If True, return all sub-classes of provided type, otherwise only search for exact match
        :return: Matching CommandSets
        """
    return [cmdset for cmdset in self._installed_command_sets if type(cmdset) == commandset_type or (subclass_match and isinstance(cmdset, commandset_type))]