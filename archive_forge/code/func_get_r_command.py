import os
from shutil import which
from .. import config
from .base import (
def get_r_command():
    if 'NIPYPE_NO_R' in os.environ:
        return None
    r_cmd = os.getenv('RCMD', default='R')
    return r_cmd if which(r_cmd) else None