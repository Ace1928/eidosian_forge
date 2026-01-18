import os
import subprocess
import sys
from typing import List, Optional, Tuple
from pip._internal.build_env import get_runnable_pip
from pip._internal.cli import cmdoptions
from pip._internal.cli.parser import ConfigOptionParser, UpdatingDefaultsHelpFormatter
from pip._internal.commands import commands_dict, get_similar_commands
from pip._internal.exceptions import CommandError
from pip._internal.utils.misc import get_pip_version, get_prog
def create_main_parser() -> ConfigOptionParser:
    """Creates and returns the main parser for pip's CLI"""
    parser = ConfigOptionParser(usage='\n%prog <command> [options]', add_help_option=False, formatter=UpdatingDefaultsHelpFormatter(), name='global', prog=get_prog())
    parser.disable_interspersed_args()
    parser.version = get_pip_version()
    gen_opts = cmdoptions.make_option_group(cmdoptions.general_group, parser)
    parser.add_option_group(gen_opts)
    parser.main = True
    description = [''] + [f'{name:27} {command_info.summary}' for name, command_info in commands_dict.items()]
    parser.description = '\n'.join(description)
    return parser