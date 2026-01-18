from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def external_logging_publisher(self):
    if self.want.external_logging_publisher is None:
        return None
    if self.have.external_logging_publisher is None and self.want.external_logging_publisher == '':
        return None
    if self.want.external_logging_publisher != self.have.external_logging_publisher:
        return self.want.external_logging_publisher