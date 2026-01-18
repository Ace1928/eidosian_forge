from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from operator import itemgetter
def get_default_ssh_config_file(self):
    return os.path.expanduser('~/.ssh/config')