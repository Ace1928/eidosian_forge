import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
def init_cfg(sys_file, platform_file, user_file, config_args=None):
    paths = get_paths_cfg(sys_file, platform_file, user_file)
    sys_config_path = paths['sys']
    platform_config_path = paths['platform']
    user_config_path = paths['user']
    cfgp = ConfigParser()
    for required in (sys_config_path, platform_config_path):
        cfgp.read([required])
    cfgp.read([user_config_path])
    if config_args is not None:
        update_cfg(cfgp, config_args)
    return cfgp