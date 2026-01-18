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
def print_env_var_values(self) -> None:
    """Get key-values pairs present as environment variables"""
    write_output('%s:', 'env_var')
    with indent_log():
        for key, value in sorted(self.configuration.get_environ_vars()):
            env_var = f'PIP_{key.upper()}'
            write_output('%s=%r', env_var, value)