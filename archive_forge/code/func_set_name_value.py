import logging
import os
import subprocess
from optparse import Values
from typing import Any, List, Optional
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
from pip._internal.exceptions import PipError
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_prog, write_output
def set_name_value(self, options: Values, args: List[str]) -> None:
    key, value = self._get_n_args(args, 'set [name] [value]', n=2)
    self.configuration.set_value(key, value)
    self._save_configuration()