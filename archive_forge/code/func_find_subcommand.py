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
def find_subcommand(action: argparse.ArgumentParser, subcmd_names: List[str]) -> argparse.ArgumentParser:
    if not subcmd_names:
        return action
    cur_subcmd = subcmd_names.pop(0)
    for sub_action in action._actions:
        if isinstance(sub_action, argparse._SubParsersAction):
            for choice_name, choice in sub_action.choices.items():
                if choice_name == cur_subcmd:
                    return find_subcommand(choice, subcmd_names)
            break
    raise CommandSetRegistrationError(f"Could not find subcommand '{full_command_name}'")