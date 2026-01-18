from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def sni_require(self):
    if self.want.sni_require is None:
        return None
    if self.want.sni_require is False:
        if self.have.sni_default is True and self.want.sni_default is None:
            raise F5ModuleError("Cannot set 'sni_require' to {0} if 'sni_default' is {1}".format(self.want.sni_require, self.have.sni_default))
    if self.want.sni_require == self.have.sni_require:
        return None
    return self.want.sni_require