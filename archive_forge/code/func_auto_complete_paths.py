import optparse
import os
import sys
from itertools import chain
from typing import Any, Iterable, List, Optional
from pip._internal.cli.main_parser import create_main_parser
from pip._internal.commands import commands_dict, create_command
from pip._internal.metadata import get_default_environment
def auto_complete_paths(current: str, completion_type: str) -> Iterable[str]:
    """If ``completion_type`` is ``file`` or ``path``, list all regular files
    and directories starting with ``current``; otherwise only list directories
    starting with ``current``.

    :param current: The word to be completed
    :param completion_type: path completion type(``file``, ``path`` or ``dir``)
    :return: A generator of regular files and/or directories
    """
    directory, filename = os.path.split(current)
    current_path = os.path.abspath(directory)
    if not os.access(current_path, os.R_OK):
        return
    filename = os.path.normcase(filename)
    file_list = (x for x in os.listdir(current_path) if os.path.normcase(x).startswith(filename))
    for f in file_list:
        opt = os.path.join(current_path, f)
        comp_file = os.path.normcase(os.path.join(directory, f))
        if completion_type != 'dir' and os.path.isfile(opt):
            yield comp_file
        elif os.path.isdir(opt):
            yield os.path.join(comp_file, '')