from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def fulldest(self):
    result = None
    if os.path.isdir(self.dest):
        result = os.path.join(self.dest, self.src)
    elif os.path.exists(os.path.dirname(self.dest)):
        result = self.dest
    else:
        try:
            os.stat(os.path.dirname(self.dest))
        except OSError as e:
            if 'permission denied' in str(e).lower():
                raise F5ModuleError('Destination directory {0} is not accessible'.format(os.path.dirname(self.dest)))
            raise F5ModuleError('Destination directory {0} does not exist'.format(os.path.dirname(self.dest)))
    if not os.access(os.path.dirname(result), os.W_OK):
        raise F5ModuleError('Destination {0} not writable'.format(os.path.dirname(result)))
    return result