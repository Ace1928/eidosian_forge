from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def source_path(self):
    if self.want.source_path is None:
        return None
    if self.want.source_path == self.have.source_path:
        if self.content:
            return self.want.source_path
    if self.want.source_path != self.have.source_path:
        return self.want.source_path