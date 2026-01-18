import optparse
import os
import sys
from itertools import chain
from typing import Any, Iterable, List, Optional
from pip._internal.cli.main_parser import create_main_parser
from pip._internal.commands import commands_dict, create_command
from pip._internal.metadata import get_default_environment
def get_path_completion_type(cwords: List[str], cword: int, opts: Iterable[Any]) -> Optional[str]:
    """Get the type of path completion (``file``, ``dir``, ``path`` or None)

    :param cwords: same as the environmental variable ``COMP_WORDS``
    :param cword: same as the environmental variable ``COMP_CWORD``
    :param opts: The available options to check
    :return: path completion type (``file``, ``dir``, ``path`` or None)
    """
    if cword < 2 or not cwords[cword - 2].startswith('-'):
        return None
    for opt in opts:
        if opt.help == optparse.SUPPRESS_HELP:
            continue
        for o in str(opt).split('/'):
            if cwords[cword - 2].split('=')[0] == o:
                if not opt.metavar or any((x in ('path', 'file', 'dir') for x in opt.metavar.split('/'))):
                    return opt.metavar
    return None