from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def encrypt_cookie_secret(self):
    if self.want.encrypt_cookie_secret != self.have.encrypt_cookie_secret:
        if self.want.update_password == 'always':
            result = self.want.encrypt_cookie_secret
            return result