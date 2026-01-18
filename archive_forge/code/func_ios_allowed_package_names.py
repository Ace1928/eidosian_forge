from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def ios_allowed_package_names(self):
    result = cmp_simple_list(self.want.ios_allowed_package_names, self.have.ios_allowed_package_names)
    return result