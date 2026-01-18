from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def check_create_dependencies(self):
    if self.want.log_publisher is None:
        if self.want.log_profile is not None:
            raise F5ModuleError("The 'log_profile' cannot be used without a defined 'log_publisher'.")