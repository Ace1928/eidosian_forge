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
def issuer_cert(self):
    if self._values['issuer_cert'] is None:
        return None
    name = fq_name(self.partition, self._values['issuer_cert'])
    true_name = flatten_boolean(self.true_names)
    if true_name == ' yes':
        return self.name
    elif name.endswith('.crt'):
        return name
    else:
        return name + '.crt'