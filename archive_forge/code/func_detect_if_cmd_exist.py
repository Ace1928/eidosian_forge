from __future__ import absolute_import, division, print_function
import shlex
import pipes
import re
import json
import os
def detect_if_cmd_exist(cmd='mongosh'):
    path = os.getenv('PATH')
    for folder in path.split(os.path.pathsep):
        mongoCmd = os.path.join(folder, cmd)
        if os.path.exists(mongoCmd) and os.access(mongoCmd, os.X_OK):
            return True
    return False