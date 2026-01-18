import logging
import os.path
from typing import List, Optional
from pip._internal.cli.spinners import open_spinner
from pip._internal.utils.setuptools_build import make_setuptools_bdist_wheel_args
from pip._internal.utils.subprocess import call_subprocess, format_command_args
def format_command_result(command_args: List[str], command_output: str) -> str:
    """Format command information for logging."""
    command_desc = format_command_args(command_args)
    text = f'Command arguments: {command_desc}\n'
    if not command_output:
        text += 'Command output: None'
    elif logger.getEffectiveLevel() > logging.DEBUG:
        text += 'Command output: [use --verbose to show]'
    else:
        if not command_output.endswith('\n'):
            command_output += '\n'
        text += f'Command output:\n{command_output}'
    return text