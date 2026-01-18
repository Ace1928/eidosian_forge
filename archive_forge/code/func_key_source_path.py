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
def key_source_path(self):
    if self.want.key_source_path is None:
        return None
    if self.want.key_source_path == self.have.key_source_path:
        if self.key_checksum:
            return self.want.key_source_path
    if self.want.key_source_path != self.have.key_source_path:
        return self.want.key_source_path