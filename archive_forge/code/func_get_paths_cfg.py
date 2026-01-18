import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
def get_paths_cfg(sys_file='pythran.cfg', platform_file='pythran-{}.cfg'.format(sys.platform), user_file='.pythranrc'):
    sys_config_dir = os.path.dirname(__file__)
    sys_config_path = os.path.join(sys_config_dir, sys_file)
    platform_config_path = os.path.join(sys_config_dir, platform_file)
    if not os.path.exists(platform_config_path):
        platform_config_path = os.path.join(sys_config_dir, 'pythran-default.cfg')
    user_config_path = os.environ.get('PYTHRANRC', None)
    if not user_config_path:
        user_config_dir = os.environ.get('XDG_CONFIG_HOME', None)
        if not user_config_dir:
            user_config_dir = os.environ.get('HOME', None)
        if not user_config_dir:
            user_config_dir = '~'
        user_config_path = os.path.expanduser(os.path.join(user_config_dir, user_file))
    return {'sys': sys_config_path, 'platform': platform_config_path, 'user': user_config_path}