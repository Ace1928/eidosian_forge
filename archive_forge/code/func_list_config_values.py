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
def list_config_values(self, options: Values, args: List[str]) -> None:
    """List config key-value pairs across different config files"""
    self._get_n_args(args, 'debug', n=0)
    self.print_env_var_values()
    for variant, files in sorted(self.configuration.iter_config_files()):
        write_output('%s:', variant)
        for fname in files:
            with indent_log():
                file_exists = os.path.exists(fname)
                write_output('%s, exists: %r', fname, file_exists)
                if file_exists:
                    self.print_config_file_values(variant)