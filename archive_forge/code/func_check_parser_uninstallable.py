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
def check_parser_uninstallable(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                attached_cmdset = getattr(subparser, constants.PARSER_ATTR_COMMANDSET, None)
                if attached_cmdset is not None and attached_cmdset is not cmdset:
                    raise CommandSetRegistrationError('Cannot uninstall CommandSet when another CommandSet depends on it')
                check_parser_uninstallable(subparser)
            break