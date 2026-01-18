import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def get_win_system_paths():
    if 'PROGRAMDATA' in os.environ:
        yield os.path.join(os.environ['PROGRAMDATA'], 'Git', 'config')
    for git_dir in _find_git_in_win_path():
        yield os.path.join(git_dir, 'etc', 'gitconfig')
    for git_dir in _find_git_in_win_reg():
        yield os.path.join(git_dir, 'etc', 'gitconfig')