import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def get_config_path(config_level: Lit_config_levels) -> str:
    if os.name == 'nt' and config_level == 'system':
        config_level = 'global'
    if config_level == 'system':
        return '/etc/gitconfig'
    elif config_level == 'user':
        config_home = os.environ.get('XDG_CONFIG_HOME') or osp.join(os.environ.get('HOME', '~'), '.config')
        return osp.normpath(osp.expanduser(osp.join(config_home, 'git', 'config')))
    elif config_level == 'global':
        return osp.normpath(osp.expanduser('~/.gitconfig'))
    elif config_level == 'repository':
        raise ValueError('No repo to get repository configuration from. Use Repo._get_config_path')
    else:
        assert_never(config_level, ValueError(f'Invalid configuration level: {config_level!r}'))