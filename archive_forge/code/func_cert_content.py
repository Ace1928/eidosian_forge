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
def cert_content(self):
    if self.want.cert_checksum != self.have.checksum:
        result = dict(checksum=self.want.cert_checksum, content=self.want.cert_content)
        return result