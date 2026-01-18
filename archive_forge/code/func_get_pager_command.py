import curses
import errno
import os
import pydoc
import subprocess
import sys
import shlex
from typing import List
def get_pager_command(default: str='less -rf') -> List[str]:
    command = shlex.split(os.environ.get('PAGER', default))
    return command