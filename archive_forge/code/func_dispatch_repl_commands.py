import click
import os
import shlex
import sys
from collections import defaultdict
from .exceptions import CommandLineParserError, ExitReplException
def dispatch_repl_commands(command):
    """
    Execute system commands entered in the repl.

    System commands are all commands starting with "!".
    """
    if command.startswith('!'):
        os.system(command[1:])
        return True
    return False